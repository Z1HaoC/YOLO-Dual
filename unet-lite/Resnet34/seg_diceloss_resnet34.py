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
torch.use_deterministic_algorithms(False) 
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
import thop
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch

# 路径配置（与YOLOv8保持一致）
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 项目根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 添加到环境变量
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 转换为相对路径

# 工具导入（统一接口）
import val as validate  # 验证与评估
from utils.callbacks import Callbacks
from utils.downloads import attempt_download
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, 
                           check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           increment_path, init_seeds, intersect_dicts, one_cycle, yaml_save, strip_optimizer)
from utils.loggers import Loggers
from utils.metrics import fitness
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, 
                              smart_DDP, smart_optimizer, smart_resume, torch_distributed_zero_first)

# 分布式训练配置（与YOLOv8保持一致）
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

# CamVid颜色映射表(RGB)用于可视化
CAMVID_COLORS = [
    [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
    [60, 40, 222], [128, 128, 0], [192, 128, 128], [64, 64, 128],
    [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 0]  # 最后一个是unlabelled
]


# -------------------------- 语义分割数据集类 --------------------------
class SegmentDataset(Dataset):
    """Camvid语义分割数据集加载类（与YOLOv8数据加载接口一致）"""
    def __init__(self, img_dir, label_dir, img_size=640, augment=False, hyp=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp or {}
        
        # 获取所有图像路径
        self.img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.img_files = [f for f in self.img_dir.iterdir() if f.suffix.lower() in self.img_extensions]
        
        # 生成对应的掩码文件路径（确保一一对应）
        self.label_files = []
        for img_file in self.img_files:
            label_file = self.label_dir / f"{img_file.stem}.png"  # 掩码与图像同名，后缀为.png
            if not label_file.exists():
                raise FileNotFoundError(f"掩码文件不存在: {label_file}")
            self.label_files.append(label_file)
        
        LOGGER.info(f"加载分割数据集: {len(self.img_files)} 张图像，{len(self.label_files)} 个掩码")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # 加载图像和掩码
        img_path = self.img_files[index]
        label_path = self.label_files[index]
        
        # 读取图像（RGB）和掩码（单通道，类别索引）
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # 转为单通道灰度图（像素值为类别索引）
        
        # 强制裁剪标签值到有效范围（12类）
        label_np = np.array(label, dtype=np.int64)
        num_classes = 12
        label_np = np.clip(label_np, 0, num_classes - 1)
        label = Image.fromarray(label_np.astype(np.uint8))
        
        # 调整尺寸并填充
        img, label = self._preprocess(img, label)
        
        # 转换为张量
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # 归一化到[0,1]
        label = torch.from_numpy(np.array(label)).long()  # 掩码保持整数类别
        
        return img, label, str(img_path)

    def _preprocess(self, img, label):
        """预处理图像和掩码（与YOLOv8预处理流程对齐）"""
        # 调整尺寸
        img, label = self._resize(img, label)
        
        # 数据增强（与YOLOv8增强接口一致）
        if self.augment and self.hyp:
            img, label = self._augment(img, label)
            
        # 填充到目标尺寸
        img, label = self._pad(img, label)
        
        return img, label
    
    def _resize(self, img, label):
        """图像缩放（与YOLOv8缩放策略一致）"""
        w, h = img.size
        scale = min(self.img_size / w, self.img_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 缩放（图像用双线性，掩码用最近邻）
        img = img.resize((new_w, new_h), Image.BILINEAR)
        label = label.resize((new_w, new_h), Image.NEAREST)
        
        return img, label
    
    def _pad(self, img, label):
        """填充图像（与YOLOv8填充策略一致）"""
        w, h = img.size
        pad_w = self.img_size - w
        pad_h = self.img_size - h
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        
        # 填充图像和掩码（使用与YOLOv8相同的填充值）
        new_img = Image.new('RGB', (self.img_size, self.img_size), (114, 114, 114))  # 与YOLOv8保持一致
        new_label = Image.new('L', (self.img_size, self.img_size), 0)  # 背景填充为0类
        
        new_img.paste(img, (pad_left, pad_top))
        new_label.paste(label, (pad_left, pad_top))
        
        return new_img, new_label
    
    def _augment(self, img, label):
        """数据增强（与YOLOv8增强接口一致，便于后续启用）"""
        # 此处保留接口，与YOLOv8增强方法对齐
        # 实际增强实现可参考YOLOv8的albumentations增强流水线
        return img, label


# -------------------------- 语义分割数据加载器 --------------------------
def create_segment_dataloader(img_dir, label_dir, img_size=640, batch_size=16, 
                             augment=False, workers=8, shuffle=True, hyp=None):
    """创建语义分割专用数据加载器（与YOLOv8数据加载器接口一致）"""
    dataset = SegmentDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        img_size=img_size,
        augment=augment,
        hyp=hyp
    )
    
    # 处理批次大小（避免超过数据集大小）
    if batch_size > len(dataset):
        batch_size = len(dataset)
        LOGGER.warning(f"批次大小调整为 {batch_size}（小于请求的批次，因数据集过小）")
    
    # 与YOLOv8保持相同的DataLoader配置
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,  # 丢弃不完整批次，确保批次大小一致
        collate_fn=lambda x: (torch.stack([i[0] for i in x]), 
                             torch.stack([i[1] for i in x]), 
                             [i[2] for i in x])
    )
    
    return dataloader, dataset


def autopad(k: int, p: Optional[int] = None) -> int:
    """自动计算卷积填充尺寸（与YOLOv8保持一致）"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """标准卷积模块: Conv2d -> BatchNorm2d -> SiLU激活（与YOLOv8一致）"""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None, 
                 g: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """融合Conv和BN用于推理加速（与YOLOv8一致）"""
        return self.act(self.conv(x))


class BasicBlock(nn.Module):
    """ResNet34专用基本残差块（与VGG16的块结构接口保持一致）"""
    expansion = 1  # 通道扩展倍数（ResNet34不扩展通道）

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, 3, stride, 1, act=True)
        self.conv2 = Conv(out_channels, out_channels, 3, 1, 1, act=False)
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return self.act(out)


class ResNet34(nn.Module):
    """ResNet34主干网络（与YOLOv8/VGG16的特征提取接口一致）"""
    def __init__(self):
        super().__init__()
        # 初始卷积层 - 仅接受3通道输入(RGB图像)
        self.in_channels = 64
        self.stem = nn.Sequential(
            Conv(3, 64, 7, 2, 3),  # 7x7卷积，输出64通道
            nn.MaxPool2d(3, 2, 1)  # 3x3池化，步长2
        )
        
        # 四个主要阶段
        self.layer1 = self._make_layer(64, 3, stride=1)    # 输出64通道
        self.layer2 = self._make_layer(128, 4, stride=2)   # 输出128通道
        self.layer3 = self._make_layer(256, 6, stride=2)   # 输出256通道
        self.layer4 = self._make_layer(512, 3, stride=2)   # 输出512通道
        
        # 记录各阶段输出用于特征金字塔（与YOLOv8特征图输出格式一致）
        self.feat_channels = [64, 128, 256]
        self.stride = torch.tensor([4, 8, 16])  # 与YOLOv8的stride定义方式一致

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        """创建ResNet的一个阶段"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            # 下采样模块用于匹配通道数和步长
            downsample = Conv(
                self.in_channels, 
                out_channels * BasicBlock.expansion, 
                1, stride, 0, act=False
            )
        
        blocks = []
        blocks.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        
        # 添加剩余的基本块
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """前向传播，返回多尺度特征（与YOLOv8特征输出格式一致）"""
        # 检查输入通道是否为3
        if x.size(1) != 3:
            raise ValueError(f"ResNet34期望输入3通道图像，实际输入{ x.size(1) }通道")
        
        # 初始处理
        x = self.stem(x)
        
        # 四个阶段的特征（与YOLOv8特征图输出顺序一致）
        f1 = self.layer1(x)  # 64通道
        f2 = self.layer2(f1) # 128通道
        f3 = self.layer3(f2) # 256通道
        
        return [f1, f2, f3]


class SegmentHead(nn.Module):
    """语义分割头（与YOLOv8分割头接口一致）"""
    def __init__(self, num_classes: int = 12, in_channels: List[int] = [64, 128, 256]):
        super().__init__()
        self.num_classes = num_classes
        
        # 特征降维与上采样（与YOLOv8的PAN-FPN结构对齐）
        self.lateral_convs = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for i, c in enumerate(in_channels):
            # 降维到相同通道数
            self.lateral_convs.append(Conv(c, 128, 1, 1))
            # 根据特征层深度设置不同的上采样倍数
            up_scale = 2 **i
            self.up_samples.append(nn.Upsample(
                scale_factor=up_scale, 
                mode='bilinear', 
                align_corners=True
            ))
        
        # 最终卷积层（与YOLOv8的检测头结构风格一致）
        self.final_conv = nn.Sequential(
            Conv(128 * 3, 256, 3, 1),
            Conv(256, num_classes, 1, 1, act=False)
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """处理多尺度特征并输出分割结果（与YOLOv8头接口一致）"""
        if len(features) != len(self.lateral_convs):
            raise ValueError(f"特征数量不匹配: 期望{len(self.lateral_convs)}个, 实际{len(features)}个")
        
        # 处理每个特征层
        processed = []
        target_size = features[0].shape[2:]  # 以第一个特征层尺寸为基准
        
        for i, (feat, lateral_conv, up_sample) in enumerate(zip(features, self.lateral_convs, self.up_samples)):
            feat = lateral_conv(feat)  # 降维
            # 上采样到目标尺寸
            if feat.shape[2:] != target_size:
                feat = up_sample(feat)
                if feat.shape[2:] != target_size:
                    feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            processed.append(feat)
        
        # 拼接特征并输出
        concat_feat = torch.cat(processed, dim=1)
        output = self.final_conv(concat_feat)
        
        return output


class ResNet34Seg(nn.Module):
    """完整的ResNet34语义分割模型（与YOLOv8/VGG16模型接口一致）"""
    def __init__(self, cfg: Dict, num_classes: Optional[int] = None):
        super().__init__()
        # 解析配置文件（与YOLOv8配置解析方式一致）
        if isinstance(cfg, str):
            cfg = check_yaml(cfg)
            with open(cfg, 'r') as f:
                self.yaml = yaml.safe_load(f)
        else:
            self.yaml = cfg
        
        # 配置参数（与YOLOv8模型属性一致）
        self.num_classes = self.yaml['nc'] if num_classes is None else num_classes
        self.yaml['nc'] = self.num_classes
        self.img_size = [640, 640]
        self.stride = torch.tensor([8, 16, 32])  # 下采样倍数（与YOLOv8定义一致）
        
        # 构建主干网络和分割头（与YOLOv8的backbone和head结构一致）
        self.backbone = ResNet34()
        self.head = SegmentHead(
            num_classes=self.num_classes,
            in_channels=self.backbone.feat_channels
        )
        
        # 初始化权重
        self._initialize_weights()
        self._log_model_info()

    def forward(self, x: torch.Tensor, augment: bool = False, 
                profile: bool = False) -> torch.Tensor:
        """前向传播入口（与YOLOv8模型forward接口一致）"""
        if augment:
            # 与YOLOv8增强推理接口一致
            return self._forward_augment(x)
        return self._forward_once(x, profile)

    def _forward_once(self, x: torch.Tensor, profile: bool = False) -> torch.Tensor:
        """单次前向传播（与YOLOv8一致的内部接口）"""
        features = self.backbone(x)
        output = self.head(features)
        
        # 上采样到输入图像尺寸
        if output.shape[2:] != x.shape[2:]:
            output = F.interpolate(
                output, 
                size=x.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        return output

    def _forward_augment(self, x: torch.Tensor) -> torch.Tensor:
        """增强推理（与YOLOv8增强推理实现一致）"""
        img_size = x.shape[-2:]
        s = [1.0, 0.83, 0.67]  # 缩放因子
        f = [None, 3, None]  # 翻转
        y = []
        
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]
            
            # 反向处理
            yi = scale_img(yi, 1 / si, gs=img_size[0])
            if fi:
                yi = yi.flip(fi)
            y.append(yi)
        
        # 融合结果
        y = torch.stack(y).mean(0)
        return y.unsqueeze(0)

    def _initialize_weights(self) -> None:
        """初始化模型权重（与YOLOv8初始化策略一致）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _log_model_info(self) -> None:
        """打印模型信息（与YOLOv8模型信息输出一致）"""
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 计算FLOPs
        input_tensor = torch.randn(1, 3, *self.img_size)
        flops, _ = thop.profile(deepcopy(self), inputs=(input_tensor,), verbose=False)
        flops_g = flops / 1e9  # 转换为GigaFLOPs
        
        LOGGER.info(f"模型信息: {n_params:,} 总参数, {n_trainable:,} 可训练参数")
        LOGGER.info(f"计算量: {flops_g:.2f} GFLOPs (输入尺寸: {self.img_size})")
        LOGGER.info(f"类别数: {self.num_classes}")


def scale_img(img: torch.Tensor, ratio: float = 1.0, gs: int = 32) -> torch.Tensor:
    """图像缩放并保持网格对齐（与YOLOv8实现一致）"""
    h, w = img.shape[2:]
    new_h = math.ceil(h * ratio / gs) * gs
    new_w = math.ceil(w * ratio / gs) * gs
    return F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)


class SegmentationLoss(nn.Module):
    """语义分割专用损失函数（与YOLOv8损失函数接口一致）"""
    def __init__(self, num_classes: int = 12, label_smoothing: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        """
        pred: 模型输出 (b, num_classes, h, w)
        target: 标签 (b, h, w) 包含类别索引
        """
        # 确保批次大小匹配
        if pred.size(0) != target.size(0):
            raise ValueError(
                f"批次大小不匹配: 模型输出批次大小为{pred.size(0)}, "
                f"目标标签批次大小为{target.size(0)}"
            )
        
        # 确保空间尺寸匹配
        if pred.shape[2:] != target.shape[1:]:
            target = F.interpolate(
                target.unsqueeze(1).float(),
                size=pred.shape[2:],
                mode='nearest'
            ).squeeze(1).long()
        
        # 计算损失（与YOLOv8损失计算方式一致）
        ce_loss = self.cross_entropy(pred, target)
        dice_loss = self._dice_loss(pred.softmax(1), self._one_hot_encode(target))
        total_loss = ce_loss + 0.5 * dice_loss
        
        return total_loss, [total_loss.item(), ce_loss.item(), dice_loss.item()]

    def _one_hot_encode(self, target: torch.Tensor) -> torch.Tensor:
        """将标签转换为one-hot编码（与YOLOv8一致）"""
        b, h, w = target.shape
        one_hot = torch.zeros(b, self.num_classes, h, w, device=target.device)
        return one_hot.scatter_(1, target.unsqueeze(1), 1.0)

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """计算Dice损失（与YOLOv8损失函数实现风格一致）"""
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + eps) / (union + eps)
        return 1.0 - dice.mean()


def seg_labels_to_class_weights(label_files, num_classes):
    """计算语义分割掩码的类别权重（与YOLOv8权重计算方式一致）"""
    class_counts = np.zeros(num_classes, dtype=np.int64)
    total = 0
    
    for file in label_files:
        # 读取掩码图像（单通道，像素值为类别索引）
        mask = np.array(Image.open(file).convert('L'), dtype=np.int64)
        # 确保掩码值在有效范围内
        if mask.max() >= num_classes:
            LOGGER.warning(f"掩码文件 {file} 包含超出类别数的索引 {mask.max()}，将被忽略")
            continue
        # 统计每个类别的像素数量
        counts = np.bincount(mask.flatten(), minlength=num_classes)
        class_counts += counts
        total += mask.size
    
    # 计算Inverse Frequency权重（与YOLOv8类别权重计算一致）
    weights = total / (num_classes * (class_counts + 1e-8))  # 加epsilon避免除零
    return torch.from_numpy(weights).float()


def mask_to_rgb(mask, color_map):
    """将类别索引掩码转换为RGB彩色掩码（与YOLOv8可视化一致）"""
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(len(color_map)):
        rgb_mask[mask == cls] = color_map[cls]
    return rgb_mask


def train(hyp: Dict, opt: argparse.Namespace, device: torch.device, callbacks: Callbacks) -> Tuple[float, ...]:
    """训练主函数（与YOLOv8训练流程一致）"""
    # 基础配置（与YOLOv8保持一致）
    save_dir = Path(opt.save_dir)
    epochs, batch_size, weights = opt.epochs, opt.batch_size, opt.weights
    resume, noval, nosave = opt.resume, opt.noval, opt.nosave
    workers, freeze = opt.workers, opt.freeze
    
    # 创建保存目录（与YOLOv8目录结构一致）
    w = save_dir / 'weights'
    w.mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'

    # 处理超参数（与YOLOv8超参数处理一致）
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)
    
    # 补充语义分割专用超参数
    hyp_defaults = {
        'lr0': 0.01, 'lrf': 0.2, 'momentum': 0.937, 'weight_decay': 0.0005,
        'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1,
        'label_smoothing': 0.0,
        # 数据增强参数（与YOLOv8保持一致的参数名）
        'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0,
        'degrees': 0.0, 'translate': 0.0, 'scale': 0.0, 'shear': 0.0,
        'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.0,
        'mosaic': 0.0, 'mixup': 0.0
    }
    
    for k, v in hyp_defaults.items():
        if k not in hyp:
            hyp[k] = v
            LOGGER.info(f"补充缺失的超参数: {k} = {v}")

    LOGGER.info(colorstr('超参数: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    yaml_save(save_dir / 'hyp.yaml', hyp)
    yaml_save(save_dir / 'opt.yaml', vars(opt))

    # 日志初始化（与YOLOv8日志系统一致）
    data_dict = None
    loggers = None
    tb_writer = None  # TensorBoard写入器
    if RANK in {-1, 0}:
        # 创建TensorBoard日志目录（与YOLOv8一致）
        experiment_dir = increment_path(save_dir / "exp", exist_ok=opt.exist_ok)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=experiment_dir)  # 初始化TensorBoard
        
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
        
        if not isinstance(callbacks, Callbacks):
            callbacks = Callbacks()
        
        # 注册回调函数（与YOLOv8回调机制一致）
        for k in dir(loggers):
            if k.startswith('on_'):
                callback = getattr(loggers, k)
                if callable(callback) and callback is not None and not k.startswith('__'):
                    try:
                        callbacks.register_action(k, callback)
                    except Exception as e:
                        LOGGER.warning(f"无法注册回调 {k}: {e}")
        
        data_dict = loggers.remote_dataset

    # 数据集配置（与YOLOv8数据加载流程一致）
    seed_value = (opt.seed + RANK) % (2**32)
    init_seeds(seed=seed_value, deterministic=False)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(opt.data)
    
    # 从数据配置中获取图像和掩码目录（与YOLOv8数据路径解析一致）
    train_img_dir = data_dict.get('train_img', data_dict.get('train', ''))
    train_label_dir = data_dict.get('train_label', os.path.join(data_dict.get('train', ''), 'labels'))
    val_img_dir = data_dict.get('val_img', data_dict.get('val', ''))
    val_label_dir = data_dict.get('val_label', os.path.join(data_dict.get('val', ''), 'labels'))
    num_classes = 1 if opt.single_cls else int(data_dict['nc'])
    names = data_dict['names']

    # 验证路径是否存在（与YOLOv8数据检查一致）
    for path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据路径不存在: {path}")
        else:
            LOGGER.info(f"找到数据路径: {path}")

    # 验证图像和掩码数量是否匹配（与YOLOv8数据验证一致）
    def count_files(path, extensions):
        count = 0
        for root, _, files in os.walk(path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    count += 1
        return count

    train_img_count = count_files(train_img_dir, ('.jpg', '.jpeg', '.png', '.bmp'))
    train_label_count = count_files(train_label_dir, ('.png',))  # 掩码都是png格式
    val_img_count = count_files(val_img_dir, ('.jpg', '.jpeg', '.png', '.bmp'))
    val_label_count = count_files(val_label_dir, ('.png',))

    LOGGER.info(f"训练集: {train_img_count} 张图像, {train_label_count} 个掩码")
    LOGGER.info(f"验证集: {val_img_count} 张图像, {val_label_count} 个掩码")

    if train_img_count != train_label_count:
        LOGGER.warning(f"训练集图像和掩码数量不匹配: {train_img_count} vs {train_label_count}")
    if val_img_count != val_label_count:
        LOGGER.warning(f"验证集图像和掩码数量不匹配: {val_img_count} vs {val_label_count}")

    # 模型初始化（与YOLOv8模型加载流程一致）
    check_suffix(weights, '.pt')
    pretrained = weights.endswith('.pt')
    model = ResNet34Seg(opt.cfg, num_classes=num_classes).to(device)
    
    # 加载预训练权重（与YOLOv8权重加载一致）
    if pretrained:
        weights = attempt_download(weights)
        ckpt = torch.load(weights, map_location='cpu')
        exclude = []
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)
        LOGGER.info(f'加载权重: {len(csd)}/{len(model.state_dict())} 项匹配')

    # 冻结层设置（与YOLOv8冻结机制一致）
    freeze = [f'backbone.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = not any(x in k for x in freeze)
        if not v.requires_grad:
            LOGGER.info(f'冻结层: {k}')

    # 图像尺寸检查（与YOLOv8尺寸检查一致）
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    # 自动批量大小（与YOLOv8自动批量机制一致）
    if RANK == -1 and batch_size == -1:
        batch_size = 16  # 回退到默认值

    # 优化器配置（与YOLOv8优化器设置一致）
    nbs = 64  # 基准批量
    accumulate = max(round(nbs / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / nbs
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # 学习率调度器（与YOLOv8调度器一致）
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA和恢复训练（与YOLOv8 EMA实现一致）
    best_fitness, start_epoch = 0.0, 0
    if pretrained and resume:
        best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, None, weights, epochs, resume)

    # 分布式配置（与YOLOv8分布式设置一致）
    cuda = device.type != 'cpu'
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # 数据加载器（与YOLOv8数据加载流程一致）
    train_loader, dataset = create_segment_dataloader(
        img_dir=train_img_dir,
        label_dir=train_label_dir,
        img_size=imgsz,
        batch_size=batch_size // WORLD_SIZE,
        augment=opt.augment,  # 由参数控制是否增强
        workers=workers,
        hyp=hyp
    )
    
    # 验证集加载器(仅主进程)
    val_loader = None
    if RANK in {-1, 0}:
        val_loader = create_segment_dataloader(
            img_dir=val_img_dir,
            label_dir=val_label_dir,
            img_size=imgsz,
            batch_size=batch_size // WORLD_SIZE * 2,
            augment=False,  # 验证集不增强
            workers=workers*2,
            shuffle=False,
            hyp=hyp
        )[0]

    # 模型属性配置（与YOLOv8模型属性一致）
    model.num_classes = num_classes
    model.hyp = hyp
    model.class_weights = seg_labels_to_class_weights(dataset.label_files, num_classes).to(device)
    model.names = names

    # 损失函数（与YOLOv8损失配置一致）
    criterion = SegmentationLoss(num_classes=num_classes, label_smoothing=hyp['label_smoothing'])
    amp = check_amp(model)
    scaler = torch.amp.GradScaler('cuda', enabled=amp) if amp else None

    stopper = EarlyStopping(patience=opt.patience)

    # 训练循环（与YOLOv8训练循环一致）
    LOGGER.info(colorstr('开始训练!'))
    t0 = time.time()
    global_step = 0  # 全局步数计数器，用于TensorBoard
    
    for epoch in range(start_epoch, epochs):
        model.train()
        mloss = torch.zeros(3, device=device)  # 损失缓存: 总损失, CE损失, Dice损失
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 6) % ('Epoch', 'gpu_mem', '总损失', 'CE损失', 'Dice损失', 'img_size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=len(train_loader), bar_format=TQDM_BAR_FORMAT)
        
        # 批量训练
        for i, (imgs, targets, paths) in pbar:
            imgs = imgs.to(device, non_blocking=True).float() # 图像归一化
            targets = targets.to(device, non_blocking=True).long()  # 掩码是(batch_size, h, w)

            # 前向传播
            with torch.amp.autocast('cuda', enabled=amp):
                pred = model(imgs)  # 预测: (b, num_classes, h, w)
                loss, loss_items = criterion(pred, targets)  # 计算损失
                loss_items = torch.tensor(loss_items, device=device)

            # 反向传播（与YOLOv8梯度处理一致）
            if amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积（与YOLOv8梯度累积一致）
            if (i + 1) % accumulate == 0 or i == len(train_loader) - 1:
                if amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # 日志更新（与YOLOv8日志输出一致）
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # 移动平均损失
                mem = f'{torch.cuda.memory_reserved() / 1E9:.3g}G' if cuda else 'N/A'
                pbar.set_description(('%10s' * 2 + '%10.4g' * 4) %
                                    (f'{epoch}/{epochs - 1}', mem, *mloss, imgs.shape[-1]))

                # TensorBoard: 记录训练损失（与YOLOv8日志一致）
                if global_step % 10 == 0:
                    tb_writer.add_scalar('Train/Total_Loss', loss_items[0], global_step)
                    tb_writer.add_scalar('Train/CE_Loss', loss_items[1], global_step)
                    tb_writer.add_scalar('Train/Dice_Loss', loss_items[2], global_step)
                    tb_writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)

                # TensorBoard: 记录图像示例（与YOLOv8可视化一致）
                if global_step % 100 == 0 and global_step != 0:
                    img_np = imgs[0].permute(1, 2, 0).cpu().detach().numpy()
                    target_np = targets[0].cpu().detach().numpy()
                    pred_np = torch.argmax(pred[0], dim=0).cpu().detach().numpy()

                    target_rgb = mask_to_rgb(target_np, CAMVID_COLORS)
                    pred_rgb = mask_to_rgb(pred_np, CAMVID_COLORS)

                    tb_writer.add_image('Train/Input_Image', img_np, global_step, dataformats='HWC')
                    tb_writer.add_image('Train/Target_Mask', target_rgb, global_step, dataformats='HWC')
                    tb_writer.add_image('Train/Predicted_Mask', pred_rgb, global_step, dataformats='HWC')

                global_step += 1

        # 学习率更新（与YOLOv8调度一致）
        scheduler.step()

        # 验证（与YOLOv8验证流程一致）
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

                # TensorBoard: 记录验证指标（与YOLOv8指标记录一致）
                tb_writer.add_scalar('Validation/mIoU', results[0], epoch)
                for cls_idx, cls_name in enumerate(names):
                    if cls_idx < len(maps):
                        tb_writer.add_scalar(f'Validation/Class_IoU/{cls_name}', maps[cls_idx], epoch)

            # 计算适应度并保存最佳模型（与YOLOv8模型保存一致）
            fi = fitness(np.array(results).reshape(1, -1))  # 适应度分数
            if fi > best_fitness:
                best_fitness = fi
            save = (not nosave) or (final_epoch and not opt.evolve)
            if save:
                torch.save({'model': ema.ema, 'optimizer': optimizer.state_dict(), 
                           'epoch': epoch, 'best_fitness': best_fitness}, last)
                if fi == best_fitness:
                    torch.save({'model': ema.ema}, best)
                    LOGGER.info(f"更新最佳模型 (mIoU: {results[0]:.4f})")
                    # 记录最佳模型预测结果
                    with torch.no_grad():
                        model.eval()
                        sample_imgs, sample_targets, _ = next(iter(val_loader))
                        sample_imgs = sample_imgs.to(device)
                        sample_preds = model(sample_imgs)
                        sample_preds = torch.argmax(sample_preds, dim=1)
                        
                        img_np = sample_imgs[0].permute(1, 2, 0).cpu().detach().numpy()
                        target_np = sample_targets[0].cpu().detach().numpy()
                        pred_np = sample_preds[0].cpu().detach().numpy()
                        
                        target_rgb = mask_to_rgb(target_np, CAMVID_COLORS)
                        pred_rgb = mask_to_rgb(pred_np, CAMVID_COLORS)
                        
                        tb_writer.add_image('Best_Model/Input_Image', img_np, epoch, dataformats='HWC')
                        tb_writer.add_image('Best_Model/Target_Mask', target_rgb, epoch, dataformats='HWC')
                        tb_writer.add_image('Best_Model/Predicted_Mask', pred_rgb, epoch, dataformats='HWC')

            # 早停检查（与YOLOv8早停机制一致）
            if stopper(epoch=epoch, fitness=fi):
                break

    # 训练结束（与YOLOv8训练结束处理一致）
    if RANK in {-1, 0}:
        if tb_writer:
            tb_writer.close()
            
        LOGGER.info(f'\n训练完成 ({(time.time() - t0) / 3600:.2f} 小时)')
        LOGGER.info(f"最佳模型保存至: {best}")
        strip_optimizer(best)  # 精简模型

    return best_fitness

import torch
torch.use_deterministic_algorithms(False)



if __name__ == '__main__':
    # 命令行参数（与YOLOv8参数保持一致）
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='初始权重路径')
    parser.add_argument('--cfg', type=str, default='resnet34.yaml', help='模型配置文件')
    parser.add_argument('--data', type=str, default='/root/BestYOLO/CamVid/data.yaml', help='数据集配置文件')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批量大小')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', default='', help='设备配置, 例如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='冻结层索引')
    parser.add_argument('--patience', type=int, default=30, help='早停 patience')
    parser.add_argument('--single-cls', action='store_true', help='单类别训练')
    parser.add_argument('--sync-bn', action='store_true', help='使用同步BN')
    parser.add_argument('--cos-lr', action='store_true', help='使用余弦学习率调度')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='恢复训练')
    parser.add_argument('--save-dir', type=str, default='runs/train-resnet34', help='结果保存目录')
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
    parser.add_argument('--hyp', type=str, default='hyp.scratch-seg.yaml', help='超参数配置文件')
    parser.add_argument('--exist-ok', action='store_true', help='允许覆盖现有实验目录')
    parser.add_argument('--augment', action='store_true', help='启用数据增强')  # 与YOLOv8保持一致的增强开关
    
    opt = parser.parse_args()
    check_requirements(exclude=('tensorboard', 'thop'))
    
    # 设备选择（与YOLOv8设备选择一致）
    device = select_device(opt.device, batch_size=opt.batch_size)
    if device.type == 'cpu':
        opt.sync_bn = False  # CPU不支持同步BN  
    # 初始化训练（与YOLOv8启动流程一致）
    callbacks = Callbacks()
    train(hyp=opt.hyp, opt=opt, device=device, callbacks=callbacks)
