# YOLOv5-ResNet50 语义分割模型训练脚本（修复版）
# 专为Camvid数据集(12类)优化，解决批次不匹配问题
import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
import thop
from torch.optim import lr_scheduler
from tqdm import tqdm

# 路径配置
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 项目根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 添加到环境变量
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 转换为相对路径

# 工具导入
import val as validate  # 验证与评估
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader  # 确保此函数返回分割标签
from utils.downloads import attempt_download
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, 
                           check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           one_cycle, yaml_save, strip_optimizer)
from utils.loggers import Loggers
from utils.metrics import fitness
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, 
                              smart_DDP, smart_optimizer, smart_resume, torch_distributed_zero_first)

# 分布式训练配置
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def autopad(k: int, p: Optional[int] = None) -> int:
    """自动计算卷积填充尺寸"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """标准卷积模块: Conv2d -> BatchNorm2d -> SiLU激活"""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None, 
                 g: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """融合Conv和BN用于推理加速"""
        return self.act(self.conv(x))


class BottleneckBlock(nn.Module):
    """ResNet50专用瓶颈块，输出通道数是输入的4倍"""
    expansion = 4  # 通道扩展倍数

    def __init__(self, in_channels: int, mid_channels: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = Conv(in_channels, mid_channels, 1, 1, 0, act=True)
        self.conv2 = Conv(mid_channels, mid_channels, 3, stride, 1, act=True)
        self.conv3 = Conv(mid_channels, mid_channels * self.expansion, 1, 1, 0, act=False)
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return self.act(out)


class ResNet50(nn.Module):
    """ResNet50主干网络（语义分割专用）"""
    def __init__(self):
        super().__init__()
        # 初始卷积层 - 仅接受3通道输入(RGB图像)
        self.in_channels = 64
        self.stem = nn.Sequential(
            Conv(3, 64, 7, 2, 3),  # 7x7卷积，输出64通道
            nn.MaxPool2d(3, 2, 1)  # 3x3池化，步长2
        )
        
        # 四个主要阶段
        self.layer1 = self._make_layer(64, 3, stride=1)   # 输出256通道
        self.layer2 = self._make_layer(128, 4, stride=2)  # 输出512通道
        self.layer3 = self._make_layer(256, 6, stride=2)  # 输出1024通道
        self.layer4 = self._make_layer(512, 3, stride=2)  # 输出2048通道
        
        # 记录各阶段输出用于特征金字塔
        self.feat_channels = [256, 512, 1024]  # 用于分割头的特征通道

    def _make_layer(self, mid_channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        """创建ResNet的一个阶段"""
        downsample = None
        if stride != 1 or self.in_channels != mid_channels * BottleneckBlock.expansion:
            # 下采样模块用于匹配通道数和步长
            downsample = Conv(
                self.in_channels, 
                mid_channels * BottleneckBlock.expansion, 
                1, stride, 0, act=False
            )
        
        blocks = []
        blocks.append(BottleneckBlock(self.in_channels, mid_channels, stride, downsample))
        self.in_channels = mid_channels * BottleneckBlock.expansion
        
        # 添加剩余的瓶颈块
        for _ in range(1, num_blocks):
            blocks.append(BottleneckBlock(self.in_channels, mid_channels))
        
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """前向传播，返回多尺度特征"""
        # 检查输入通道是否为3
        if x.size(1) != 3:
            raise ValueError(f"ResNet50期望输入3通道图像，实际输入{ x.size(1) }通道")
        
        # 初始处理
        x = self.stem(x)
        
        # 四个阶段的特征（返回前三个用于分割）
        f1 = self.layer1(x)  # 256通道 (320x320)
        f2 = self.layer2(f1) # 512通道 (160x160)
        f3 = self.layer3(f2) # 1024通道 (80x80)
        
        return [f1, f2, f3]


class SegmentHead(nn.Module):
    """语义分割头，处理多尺度特征并输出像素级分类结果"""
    def __init__(self, num_classes: int = 12, in_channels: List[int] = [256, 512, 1024]):
        super().__init__()
        self.num_classes = num_classes
        
        # 特征降维与上采样
        self.lateral_convs = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for i, c in enumerate(in_channels):
            # 降维到相同通道数（128）
            self.lateral_convs.append(Conv(c, 128, 1, 1))
            # 上采样倍数：第0层(320x320)不需要上采样，第1层(160x160)需2倍，第2层(80x80)需4倍
            up_scale = 2 **i
            self.up_samples.append(nn.Upsample(
                scale_factor=up_scale, 
                mode='bilinear', 
                align_corners=True
            ))
        
        # 最终卷积层（融合特征并输出类别）
        self.final_conv = nn.Sequential(
            Conv(128 * 3, 256, 3, 1),
            Conv(256, num_classes, 1, 1, act=False)  # 输出通道=类别数
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """处理多尺度特征并输出分割结果"""
        if len(features) != len(self.lateral_convs):
            raise ValueError(f"特征数量不匹配: 期望{len(self.lateral_convs)}个, 实际{len(features)}个")
        
        # 处理每个特征层
        processed = []
        target_size = features[0].shape[2:]  # 以第一个特征层尺寸(320x320)为基准
        
        for i, (feat, lateral_conv, up_sample) in enumerate(zip(features, self.lateral_convs, self.up_samples)):
            feat = lateral_conv(feat)  # 降维到128通道
            # 上采样到目标尺寸（处理可能的舍入误差）
            if feat.shape[2:] != target_size:
                feat = up_sample(feat)
                if feat.shape[2:] != target_size:
                    feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            processed.append(feat)
        
        # 拼接特征并输出
        concat_feat = torch.cat(processed, dim=1)  # 128*3=384通道
        output = self.final_conv(concat_feat)  # 输出(12, 320, 320)
        
        # 上采样到输入图像尺寸(640x640)
        output = F.interpolate(output, size=(640, 640), mode='bilinear', align_corners=False)
        return output


class ResNet50Seg(nn.Module):
    """完整的ResNet50语义分割模型（输出像素级分类）"""
    def __init__(self, cfg: Dict, num_classes: Optional[int] = None):
        super().__init__()
        # 解析配置文件
        if isinstance(cfg, str):
            cfg = check_yaml(cfg)
            with open(cfg, 'r') as f:
                self.yaml = yaml.safe_load(f)
        else:
            self.yaml = cfg
        
        # 配置参数
        self.num_classes = self.yaml['nc'] if num_classes is None else num_classes
        self.yaml['nc'] = self.num_classes
        self.img_size = [640, 640]
        self.stride = torch.tensor([8, 16, 32])  # 下采样倍数
        
        # 构建主干网络和分割头
        self.backbone = ResNet50()
        self.head = SegmentHead(
            num_classes=self.num_classes,
            in_channels=self.backbone.feat_channels
        )
        
        # 初始化权重
        self._initialize_weights()
        self._log_model_info()

    def forward(self, x: torch.Tensor, augment: bool = False, 
                profile: bool = False) -> torch.Tensor:
        """前向传播入口"""
        if augment:
            return self._forward_augment(x)
        return self._forward_once(x, profile)

    def _forward_once(self, x: torch.Tensor, profile: bool = False) -> torch.Tensor:
        """单次前向传播（输出：(batch_size, num_classes, 640, 640)）"""
        features = self.backbone(x)  # 多尺度特征
        output = self.head(features)  # 分割结果
        return output

    def _forward_augment(self, x: torch.Tensor) -> torch.Tensor:
        """数据增强模式（多尺度+翻转）"""
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]  # 缩放因子
        f = [None, 3, None]  # 翻转模式（3=左右翻转）
        outputs = []
        
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)
            yi = F.interpolate(yi, size=img_size, mode='bilinear', align_corners=False)
            if fi == 3:
                yi = yi.flip(3)  # 恢复翻转
            outputs.append(yi)
        
        return torch.mean(torch.stack(outputs), dim=0)  # 平均多尺度结果

    def _initialize_weights(self) -> None:
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _log_model_info(self) -> None:
        """打印模型信息"""
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        LOGGER.info(f"模型信息: {n_params:,} 总参数, {n_trainable:,} 可训练参数")
        LOGGER.info(f"类别数: {self.num_classes}")


class SegmentationLoss(nn.Module):
    """语义分割专用损失函数（交叉熵+Dice损失）"""
    def __init__(self, num_classes: int = 12, label_smoothing: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        """
        pred: 模型输出 (batch_size, num_classes, h, w)
        target: 标签 (batch_size, h, w) 包含类别索引（0~11）
        """
        # 检查批次大小是否匹配
        if pred.size(0) != target.size(0):
            raise ValueError(
                f"批次大小不匹配: 模型输出批次{pred.size(0)}, 目标标签批次{target.size(0)}"
            )
        
        # 调整目标尺寸以匹配预测（若有差异）
        if pred.shape[2:] != target.shape[1:]:
            target = F.interpolate(
                target.unsqueeze(1).float(),  # 临时添加通道维度
                size=pred.shape[2:],
                mode='nearest'  # 标签用最近邻插值
            ).squeeze(1).long()  # 恢复形状并转为长整数
        
        # 计算交叉熵损失（自动处理通道维度）
        ce_loss = self.cross_entropy(pred, target)
        
        # 计算Dice损失（需将标签转为one-hot）
        dice_loss = self._dice_loss(pred.softmax(1), self._one_hot_encode(target))
        
        # 总损失（交叉熵为主，Dice辅助）
        total_loss = ce_loss + 0.5 * dice_loss
        return total_loss, [total_loss.item(), ce_loss.item(), dice_loss.item()]

    def _one_hot_encode(self, target: torch.Tensor) -> torch.Tensor:
        """将标签转为one-hot编码 (batch_size, num_classes, h, w)"""
        b, h, w = target.shape
        one_hot = torch.zeros(b, self.num_classes, h, w, device=target.device)
        return one_hot.scatter_(1, target.unsqueeze(1), 1.0)  # 在通道维度上散射

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Dice损失：衡量预测与目标的重叠度"""
        intersection = (pred * target).sum(dim=(2, 3))  # 交
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))  # 并
        dice = (2.0 * intersection + eps) / (union + eps)  # Dice系数
        return 1.0 - dice.mean()  # 损失=1-平均系数


def scale_img(img: torch.Tensor, ratio: float = 1.0, gs: int = 32) -> torch.Tensor:
    """图像缩放并保持网格对齐"""
    h, w = img.shape[2:]
    new_h = math.ceil(h * ratio / gs) * gs
    new_w = math.ceil(w * ratio / gs) * gs
    return F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)


def train(hyp: Dict, opt: argparse.Namespace, device: torch.device, callbacks: Callbacks) -> Tuple[float, ...]:
    """训练主函数（修复标签处理逻辑）"""
    # 基础配置
    save_dir = Path(opt.save_dir)
    epochs, batch_size, weights = opt.epochs, opt.batch_size, opt.weights
    resume, noval, nosave = opt.resume, opt.noval, opt.nosave
    workers, freeze = opt.workers, opt.freeze
    
    # 创建保存目录
    w = save_dir / 'weights'
    w.mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'

    # 处理超参数（补充语义分割专用参数）
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)
    
    hyp_defaults = {
        'lr0': 0.01, 'lrf': 0.2, 'momentum': 0.937, 'weight_decay': 0.0005,
        'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1,
        'label_smoothing': 0.0,  # 标签平滑
        'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,  # 色彩增强
        'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0,  # 几何增强
        'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5,
        'mosaic': 1.0, 'mixup': 0.0
    }
    
    for k, v in hyp_defaults.items():
        if k not in hyp:
            hyp[k] = v
            LOGGER.info(f"补充缺失的超参数: {k} = {v}")

    LOGGER.info(colorstr('超参数: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    yaml_save(save_dir / 'hyp.yaml', hyp)
    yaml_save(save_dir / 'opt.yaml', vars(opt))

    # 日志初始化
    data_dict = None
    loggers = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
        
        if not isinstance(callbacks, Callbacks):
            callbacks = Callbacks()
        
        # 注册回调函数
        for k in dir(loggers):
            if k.startswith('on_') and callable(getattr(loggers, k)):
                try:
                    callbacks.register_action(k, getattr(loggers, k))
                except Exception as e:
                    LOGGER.warning(f"无法注册回调 {k}: {e}")
        
        data_dict = loggers.remote_dataset

    # 数据集配置（确保加载分割标签）
    seed_value = (opt.seed + RANK) % (2**32)
    init_seeds(seed=seed_value, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(opt.data)
    train_path, val_path = data_dict['train'], data_dict['val']
    num_classes = 1 if opt.single_cls else int(data_dict['nc'])
    names = data_dict['names']

    # 模型初始化（语义分割模型）
    check_suffix(weights, '.pt')
    pretrained = weights.endswith('.pt')
    model = ResNet50Seg(opt.cfg, num_classes=num_classes).to(device)
    
    # 加载预训练权重
    if pretrained:
        weights = attempt_download(weights)
        ckpt = torch.load(weights, map_location='cpu')
        exclude = []  # 不排除任何层（分割模型权重兼容）
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)
        LOGGER.info(f'加载权重: {len(csd)}/{len(model.state_dict())} 项匹配')

    # 冻结层设置（仅冻结backbone前几层）
    freeze = [f'backbone.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = not any(x in k for x in freeze)
        if not v.requires_grad:
            LOGGER.info(f'冻结层: {k}')

    # 图像尺寸检查
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    # 自动批量大小
    if RANK == -1 and batch_size == -1:
        batch_size = check_train_batch_size(model, imgsz, check_amp(model))

    # 优化器配置
    nbs = 64  # 基准批量
    accumulate = max(round(nbs / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / nbs
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # 学习率调度器
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA和恢复训练
    ema = ModelEMA(model) if RANK in {-1, 0} else None
    best_fitness, start_epoch = 0.0, 0
    if pretrained and resume:
        best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)

    # 分布式配置
    cuda = device.type != 'cpu'
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 数据加载器（关键：确保返回的targets是(batch_size, h, w)的分割标签）
    train_loader, dataset = create_dataloader(
        train_path, imgsz, batch_size // WORLD_SIZE, gs, opt.single_cls, hyp=hyp,
        augment=True, cache=opt.cache, rect=opt.rect, rank=LOCAL_RANK, workers=workers,
        prefix=colorstr('train: '))
    
    # 验证集加载器(仅主进程)
    val_loader = None
    if RANK in {-1, 0}:
        val_loader = create_dataloader(
            val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, opt.single_cls,
            hyp=hyp, cache=opt.cache, rect=True, rank=-1, workers=workers*2,
            prefix=colorstr('val: '))[0]

    # 模型属性配置
    model.num_classes = num_classes
    model.hyp = hyp
    model.class_weights = labels_to_class_weights(dataset.labels, num_classes).to(device) * num_classes
    model.names = names

    # 损失函数（使用语义分割专用损失）
    criterion = SegmentationLoss(num_classes=num_classes, label_smoothing=hyp['label_smoothing'])
    amp = check_amp(model)
    # 修复PyTorch 2.0+的amp接口变更
    scaler = torch.amp.GradScaler('cuda', enabled=amp) if amp else None

    stopper = EarlyStopping(patience=opt.patience)

    # 训练循环（核心修复：标签处理）
    LOGGER.info(colorstr('开始训练!'))
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        mloss = torch.zeros(3, device=device)  # 损失缓存: 总损失, CE损失, Dice损失
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 6) % ('Epoch', 'gpu_mem', '总损失', 'CE损失', 'Dice损失', 'img_size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=len(train_loader), bar_format=TQDM_BAR_FORMAT)
        
        # 批量训练（修复标签形状处理）
        for i, (imgs, targets, paths, _) in pbar:
            # 图像预处理：归一化到[0,1]
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            
            # 标签处理：确保形状为(batch_size, h, w)，无多余维度
            # 若原始标签带通道维度(如(batch_size, 1, h, w))，则挤压通道维度
            if targets.dim() == 4 and targets.size(1) == 1:
                targets = targets.squeeze(1)  # 转为(batch_size, h, w)
            targets = targets.to(device, non_blocking=True).long()  # 确保是长整数类型

            # 前向传播（使用新版amp接口）
            with torch.amp.autocast('cuda', enabled=amp):
                pred = model(imgs)  # 模型输出: (batch_size, 12, 640, 640)
                loss, loss_items = criterion(pred, targets)  # 计算损失（批次必须匹配）

            # 反向传播
            if amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积
            if (i + 1) % accumulate == 0 or i == len(train_loader) - 1:
                if amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # 日志更新
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # 移动平均损失
                mem = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'  # 显存
                pbar.set_description(('%10s' * 2 + '%10.4g' * 4) %
                                    (f'{epoch}/{epochs - 1}', mem, *mloss, imgs.shape[-1]))

        # 学习率更新
        scheduler.step()

        # 验证
        if RANK in {-1, 0}:
            # 保存模型
            ema.update_attr(model, include=['yaml', 'num_classes', 'hyp', 'names', 'stride'])
            final_epoch = (epoch == epochs - 1)
            if not noval or final_epoch:
                results, maps, _ = validate.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    model=ema.ema,
                    single_cls=opt.single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    verbose=True
                )

            # 计算适应度并保存最佳模型
            fi = fitness(np.array(results).reshape(1, -1))  # 适应度分数
            if fi > best_fitness:
                best_fitness = fi
            save = (not nosave) or (final_epoch and not opt.evolve)
            if save:
                torch.save({'model': ema.ema, 'optimizer': optimizer.state_dict(), 
                           'epoch': epoch, 'best_fitness': best_fitness}, last)
                if fi == best_fitness:
                    torch.save({'model': ema.ema}, best)

            # 早停检查
            if stopper(epoch=epoch, fitness=fi):
                break

    # 训练结束
    if RANK in {-1, 0}:
        LOGGER.info(f'\n训练完成 ({(time.time() - t0) / 3600:.2f} 小时)')
        LOGGER.info(f"最佳模型保存至: {best}")
        strip_optimizer(best)  # 精简模型

    return best_fitness


if __name__ == '__main__':
    # 命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='初始权重路径')
    parser.add_argument('--cfg', type=str, default='models/resnet50.yaml', help='模型配置文件')
    parser.add_argument('--data', type=str, default='Camvid/data.yaml', help='数据集配置文件')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批量大小')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', default='', help='设备配置, 例如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='冻结层索引')
    parser.add_argument('--patience', type=int, default=10, help='早停 patience')
    parser.add_argument('--single-cls', action='store_true', help='单类别训练')
    parser.add_argument('--sync-bn', action='store_true', help='使用同步BN')
    parser.add_argument('--cos-lr', action='store_true', help='使用余弦学习率调度')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='恢复训练')
    parser.add_argument('--save-dir', type=str, default='runs/train-seg', help='结果保存目录')
    parser.add_argument('--optimizer', type=str, default='SGD', help='优化器类型')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='标签平滑因子')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--noval', action='store_true', help='禁用验证')
    parser.add_argument('--nosave', action='store_true', help='不保存模型')
    parser.add_argument('--noplots', action='store_true', help='不生成可视化图表')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='缓存图像到内存/磁盘')
    parser.add_argument('--rect', action='store_true', help='使用矩形训练')
    parser.add_argument('--image-weights', action='store_true', help='使用图像权重采样')
    parser.add_argument('--quad', action='store_true', help='使用四分之一采样')
    parser.add_argument('--evolve', action='store_true', help='进化超参数')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-seg.yaml', 
                       help='超参数配置文件')
    
    opt = parser.parse_args()
    check_requirements(exclude=('tensorboard', 'thop'))
    
    # 设备选择
    device = select_device(opt.device, batch_size=opt.batch_size)
    if device.type == 'cpu':
        opt.sync_bn = False  # CPU不支持同步BN  
    
    # 初始化训练
    callbacks = Callbacks()
    train(hyp=opt.hyp, opt=opt, device=device, callbacks=callbacks)