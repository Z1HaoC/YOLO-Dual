import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import json

import numpy as np
import torch
torch.use_deterministic_algorithms(False) 
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
import thop
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # 导入TensorBoard
import matplotlib.pyplot as plt

# 路径配置
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 项目根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 添加到环境变量
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 转换为相对路径

# 工具导入
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

# 分布式训练配置
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

# CamVid颜色映射表(RGB)用于可视化
CAMVID_COLORS = [
    [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
    [60, 40, 222], [128, 128, 0], [192, 128, 128], [64, 64, 128],
    [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 0]  # 最后一个是unlabelled
]

# 类别名称
CLASS_NAMES = [
    "sky", "building", "pole", "road", "pavement",
    "tree", "signsymbol", "fence", "car", "pedestrian",
    "bicyclist", "unlabelled"
]
NUM_CLASSES = len(CLASS_NAMES)


# -------------------------- 数据增强工具函数 --------------------------
class RandomHorizontalFlip:
    """随机水平翻转"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        return img, mask


class RandomVerticalFlip:
    """随机垂直翻转"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        return img, mask


class RandomRotation:
    """随机旋转"""
    def __init__(self, degrees=15, p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            img = img.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle, resample=Image.NEAREST)
        return img, mask


class RandomBrightness:
    """随机调整亮度"""
    def __init__(self, factor_range=(0.7, 1.3), p=0.5):
        self.factor_range = factor_range
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            factor = random.uniform(*self.factor_range)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        return img, mask


class RandomContrast:
    """随机调整对比度"""
    def __init__(self, factor_range=(0.7, 1.3), p=0.5):
        self.factor_range = factor_range
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            factor = random.uniform(*self.factor_range)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)
        return img, mask


class RandomGaussianBlur:
    """随机高斯模糊"""
    def __init__(self, radius_range=(0.5, 2.0), p=0.2):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img, mask


class RandomCrop:
    """随机裁剪"""
    def __init__(self, scale_range=(0.7, 1.0), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            w, h = img.size
            scale = random.uniform(*self.scale_range)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 确保新尺寸至少为1
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            
            # 随机选择裁剪区域
            x1 = random.randint(0, w - new_w)
            y1 = random.randint(0, h - new_h)
            
            img = img.crop((x1, y1, x1 + new_w, y1 + new_h))
            mask = mask.crop((x1, y1, x1 + new_w, y1 + new_h))
            
            # 调整回原始尺寸
            img = img.resize((w, h), Image.BILINEAR)
            mask = mask.resize((w, h), Image.NEAREST)
            
        return img, mask


def get_augmentations(hyp):
    """根据超参数获取数据增强组合"""
    augmentations = [
        RandomHorizontalFlip(p=hyp.get('fliplr', 0.5)),
        RandomVerticalFlip(p=hyp.get('flipud', 0.2)),
        RandomRotation(degrees=hyp.get('degrees', 15), p=0.3),
        RandomBrightness(factor_range=(0.7, 1.3), p=0.3),
        RandomContrast(factor_range=(0.7, 1.3), p=0.3),
        RandomGaussianBlur(p=0.1),
        RandomCrop(scale_range=(0.7, 1.0), p=0.3)
    ]
    return augmentations


# -------------------------- 掩码转JSON功能 --------------------------
def mask_to_json(mask_path, json_save_path):
    """将单张掩码图像转换为JSON格式"""
    try:
        # 读取掩码并转换为数组
        mask = Image.open(mask_path).convert("L")  # 单通道灰度图
        mask_np = np.array(mask, dtype=np.int64)
        
        # 裁剪异常值（确保在有效类别范围内）
        mask_np = np.clip(mask_np, 0, NUM_CLASSES - 1)
        
        # 构建JSON数据
        json_data = {
            "filename": mask_path.name,
            "shape": mask_np.shape,  # (高, 宽)
            "dtype": str(mask_np.dtype),
            "class_names": CLASS_NAMES,
            "mask_data": mask_np.flatten().tolist()  # 展平为一维列表存储
        }
        
        # 保存为JSON
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        LOGGER.warning(f"转换 {mask_path} 失败: {str(e)}")
        return False

def batch_convert_masks_to_json(mask_dir, json_save_dir):
    """批量将掩码图像转换为JSON格式"""
    os.makedirs(json_save_dir, exist_ok=True)
    
    mask_paths = list(Path(mask_dir).glob("*.png"))
    if not mask_paths:
        LOGGER.warning(f"在 {mask_dir} 中未找到任何PNG掩码文件")
        return 0, 0
    
    success_count = 0
    for mask_path in tqdm(mask_paths, desc="转换掩码为JSON"):
        json_filename = mask_path.stem + ".json"
        json_save_path = os.path.join(json_save_dir, json_filename)
        if mask_to_json(mask_path, json_save_path):
            success_count += 1
    
    LOGGER.info(f"转换完成: {success_count}/{len(mask_paths)} 个掩码成功转换")
    LOGGER.info(f"JSON文件保存至: {json_save_dir}")
    return success_count, len(mask_paths)

def verify_json_masks(img_dir, json_mask_dir):
    """验证图像和JSON掩码是否匹配"""
    # 获取所有图像文件
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    img_files = [f for f in Path(img_dir).iterdir() if f.suffix.lower() in img_extensions]
    
    if not img_files:
        LOGGER.warning(f"在 {img_dir} 中未找到任何图像文件")
        return False
    
    # 检查每个图像是否有对应的JSON掩码
    missing = []
    for img_file in img_files:
        json_file = Path(json_mask_dir) / f"{img_file.stem}.json"
        if not json_file.exists():
            missing.append(img_file.name)
    
    if missing:
        LOGGER.warning(f"发现 {len(missing)} 个图像缺少对应的JSON掩码:")
        for name in missing[:10]:  # 只显示前10个
            LOGGER.warning(f"  - {name}")
        if len(missing) > 10:
            LOGGER.warning(f"  ... 还有 {len(missing)-10} 个缺失文件")
        return False
    else:
        LOGGER.info(f"所有 {len(img_files)} 个图像都有对应的JSON掩码")
        return True


# -------------------------- 语义分割数据集类（JSON版本） --------------------------
class JSONSegmentDataset(Dataset):
    """从JSON文件加载掩码的语义分割数据集类，支持数据增强"""
    def __init__(self, img_dir, json_label_dir, img_size=640, augment=False, hyp=None):
        self.img_dir = Path(img_dir)
        self.json_label_dir = Path(json_label_dir)
        self.img_size = img_size
        self.augment = augment
        
        # 获取数据增强方法
        self.augmentations = get_augmentations(hyp) if (augment and hyp is not None) else []
        
        # 获取所有图像路径
        self.img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.img_files = [f for f in self.img_dir.iterdir() if f.suffix.lower() in self.img_extensions]
        
        if not self.img_files:
            raise FileNotFoundError(f"在 {img_dir} 中未找到任何图像文件")
        
        # 生成对应的JSON掩码文件路径（确保一一对应）
        self.json_files = []
        for img_file in self.img_files:
            json_file = self.json_label_dir / f"{img_file.stem}.json"
            if not json_file.exists():
                raise FileNotFoundError(f"JSON掩码文件不存在: {json_file}")
            self.json_files.append(json_file)
        
        # 从第一个JSON文件获取类别信息
        with open(self.json_files[0], "r") as f:
            first_json = json.load(f)
        self.class_names = first_json["class_names"]
        self.num_classes = len(self.class_names)
        
        LOGGER.info(f"加载JSON分割数据集: {len(self.img_files)} 张图像，{len(self.json_files)} 个JSON掩码，数据增强: {augment}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # 加载图像和JSON掩码
        img_path = self.img_files[index]
        json_path = self.json_files[index]
        
        # 读取图像（RGB）
        img = Image.open(img_path).convert('RGB')
        
        # 从JSON加载掩码数据
        with open(json_path, "r") as f:
            json_data = json.load(f)
        
        # 解析掩码数据（从一维列表恢复为二维数组）
        mask_shape = json_data["shape"]  # (高, 宽)
        mask_flat = np.array(json_data["mask_data"], dtype=np.int64)
        mask_np = mask_flat.reshape(mask_shape)  # 恢复形状
        
        # 强制裁剪标签值到有效范围
        mask_np = np.clip(mask_np, 0, self.num_classes - 1)
        mask = Image.fromarray(mask_np.astype(np.uint8))
        
        # 应用数据增强
        if self.augment and self.augmentations:
            img, mask = self._apply_augmentations(img, mask)
        
        # 调整尺寸并填充
        img, mask = self._resize_and_pad(img, mask)
        
        # 转换为张量
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # 归一化到[0,1]
        mask = torch.from_numpy(np.array(mask)).long()  # 掩码保持整数类别
        
        return img, mask, str(img_path)

    def _apply_augmentations(self, img, mask):
        """应用数据增强"""
        # 随机打乱增强顺序
        augmented = random.sample(self.augmentations, k=len(self.augmentations))
        
        for aug in augmented:
            img, mask = aug(img, mask)
        return img, mask

    def _resize_and_pad(self, img, mask):
        """调整图像和掩码尺寸并填充至目标大小"""
        w, h = img.size
        scale = min(self.img_size / w, self.img_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 缩放（图像用双线性，掩码用最近邻）
        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        
        # 计算填充
        pad_w = self.img_size - new_w
        pad_h = self.img_size - new_h
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        
        # 填充图像和掩码
        new_img = Image.new('RGB', (self.img_size, self.img_size), (128, 128, 128))  # 灰色填充
        new_label = Image.new('L', (self.img_size, self.img_size), 0)  # 背景填充为0类
        
        new_img.paste(img, (pad_left, pad_top))
        new_label.paste(mask, (pad_left, pad_top))
        
        return new_img, new_label


# -------------------------- 语义分割数据加载器（JSON版本） --------------------------
def create_json_segment_dataloader(img_dir, json_label_dir, img_size=640, batch_size=16, 
                                  augment=False, workers=8, shuffle=True, hyp=None):
    """创建从JSON标签加载数据的语义分割专用数据加载器"""
    dataset = JSONSegmentDataset(
        img_dir=img_dir,
        json_label_dir=json_label_dir,
        img_size=img_size,
        augment=augment,
        hyp=hyp
    )
    
    # 处理批次大小（避免超过数据集大小）
    if batch_size > len(dataset):
        batch_size = len(dataset)
        LOGGER.warning(f"批次大小调整为 {batch_size}（小于请求的批次，因数据集过小）")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=True  # 丢弃不完整批次，确保批次大小一致
    )
    
    return dataloader, dataset


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


class BasicBlock(nn.Module):
    """ResNet34专用基本残差块，无通道扩展"""
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
    """ResNet34主干网络"""
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
        
        # 记录各阶段输出用于特征金字塔
        self.feat_channels = [64, 128, 256]

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
        """前向传播，返回多尺度特征"""
        # 检查输入通道是否为3
        if x.size(1) != 3:
            raise ValueError(f"ResNet34期望输入3通道图像，实际输入{ x.size(1) }通道")
        
        # 初始处理
        x = self.stem(x)
        
        # 四个阶段的特征
        f1 = self.layer1(x)  # 64通道
        f2 = self.layer2(f1) # 128通道
        f3 = self.layer3(f2) # 256通道
        
        return [f1, f2, f3]


class SegmentHead(nn.Module):
    """语义分割头，处理多尺度特征"""
    def __init__(self, num_classes: int = 12, in_channels: List[int] = [64, 128, 256]):
        super().__init__()
        self.num_classes = num_classes
        
        # 特征降维与上采样
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
        
        # 最终卷积层
        self.final_conv = nn.Sequential(
            Conv(128 * 3, 256, 3, 1),
            Conv(256, num_classes, 1, 1, act=False)
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """处理多尺度特征并输出分割结果"""
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
    """完整的ResNet34语义分割模型"""
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
        """前向传播入口"""
        return self._forward_once(x, profile)

    def _forward_once(self, x: torch.Tensor, profile: bool = False) -> torch.Tensor:
        """单次前向传播"""
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
        
        # 计算FLOPs
        input_tensor = torch.randn(1, 3, *self.img_size)
        flops, _ = thop.profile(deepcopy(self), inputs=(input_tensor,), verbose=False)
        flops_g = flops / 1e9  # 转换为GigaFLOPs
        
        LOGGER.info(f"模型信息: {n_params:,} 总参数, {n_trainable:,} 可训练参数")
        LOGGER.info(f"计算量: {flops_g:.2f} GFLOPs (输入尺寸: {self.img_size})")
        LOGGER.info(f"类别数: {self.num_classes}")


def scale_img(img: torch.Tensor, ratio: float = 1.0, gs: int = 32) -> torch.Tensor:
    """图像缩放并保持网格对齐"""
    h, w = img.shape[2:]
    new_h = math.ceil(h * ratio / gs) * gs
    new_w = math.ceil(w * ratio / gs) * gs
    return F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)


class SegmentationLoss(nn.Module):
    """语义分割专用损失函数，支持交叉熵+Dice损失"""
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
        
        # 计算损失
        ce_loss = self.cross_entropy(pred, target)
        dice_loss = self._dice_loss(pred.softmax(1), self._one_hot_encode(target))
        total_loss = ce_loss + 0.5 * dice_loss
        
        return total_loss, [total_loss.item(), ce_loss.item(), dice_loss.item()]

    def _one_hot_encode(self, target: torch.Tensor) -> torch.Tensor:
        """将标签转换为one-hot编码"""
        b, h, w = target.shape
        one_hot = torch.zeros(b, self.num_classes, h, w, device=target.device)
        return one_hot.scatter_(1, target.unsqueeze(1), 1.0)

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """计算Dice损失"""
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + eps) / (union + eps)
        return 1.0 - dice.mean()


def seg_labels_to_class_weights(json_files, num_classes):
    """从JSON文件计算语义分割掩码的类别权重（Inverse Frequency）"""
    class_counts = np.zeros(num_classes, dtype=np.int64)
    total = 0
    
    for file in json_files:
        try:
            # 从JSON加载掩码数据
            with open(file, "r") as f:
                json_data = json.load(f)
            
            # 解析掩码数据
            mask_flat = np.array(json_data["mask_data"], dtype=np.int64)
            
            # 统计每个类别的像素数量
            counts = np.bincount(mask_flat, minlength=num_classes)
            class_counts += counts
            total += len(mask_flat)
        except Exception as e:
            LOGGER.warning(f"处理JSON文件 {file} 时出错: {str(e)}")
            continue
    
    # 计算Inverse Frequency权重
    weights = total / (num_classes * (class_counts + 1e-8))  # 加epsilon避免除零
    return torch.from_numpy(weights).float()


def mask_to_rgb(mask, color_map):
    """将类别索引掩码转换为RGB彩色掩码"""
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(len(color_map)):
        rgb_mask[mask == cls] = color_map[cls]
    return rgb_mask


def visualize_prediction_difference(gt_mask, pred_mask, color_map):
    """可视化预测与真实掩码的差异区域"""
    h, w = gt_mask.shape
    diff_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 找到差异区域
    diff_regions = gt_mask != pred_mask
    
    # 为差异区域上色（使用洋红色标记）
    diff_mask[diff_regions] = [255, 0, 255]  # 洋红色
    
    # 叠加真实掩码和预测掩码的颜色
    for cls in range(len(color_map)):
        # 真实掩码中正确预测的区域
        correct_regions = (gt_mask == cls) & (pred_mask == cls)
        diff_mask[correct_regions] = color_map[cls]
    
    return diff_mask


def train(hyp: Dict, opt: argparse.Namespace, device: torch.device, callbacks: Callbacks) -> Tuple[float, ...]:
    """训练主函数（JSON版本，带数据增强）"""
    # 基础配置
    save_dir = Path(opt.save_dir)
    epochs, batch_size, weights = opt.epochs, opt.batch_size, opt.weights
    resume, noval, nosave = opt.resume, opt.noval, opt.nosave
    workers, freeze = opt.workers, opt.freeze
    
    # 创建保存目录
    w = save_dir / 'weights'
    w.mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'

    # 处理超参数
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)
    
    # 补充语义分割专用超参数
    hyp_defaults = {
        'lr0': 0.01, 'lrf': 0.2, 'momentum': 0.937, 'weight_decay': 0.0005,
        'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1,
        'label_smoothing': 0.0,
        # 数据增强参数
        'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
        'degrees': 10.0, 'translate': 0.1, 'scale': 0.5, 'shear': 2.0,
        'perspective': 0.001, 'flipud': 0.2, 'fliplr': 0.5
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
    tb_writer = None  # TensorBoard写入器
    if RANK in {-1, 0}:
        # 创建TensorBoard日志目录
        experiment_dir = increment_path(save_dir / "exp", exist_ok=opt.exist_ok)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=experiment_dir)  # 初始化TensorBoard
        
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
        
        if not isinstance(callbacks, Callbacks):
            callbacks = Callbacks()
        
        # 注册回调函数
        for k in dir(loggers):
            if k.startswith('on_'):
                callback = getattr(loggers, k)
                if callable(callback) and callback is not None and not k.startswith('__'):
                    try:
                        callbacks.register_action(k, callback)
                    except Exception as e:
                        LOGGER.warning(f"无法注册回调 {k}: {e}")
        
        data_dict = loggers.remote_dataset

    # 数据集配置
    seed_value = (opt.seed + RANK) % (2**32)
    init_seeds(seed=seed_value, deterministic=False)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(opt.data)
    
    # 从数据配置中获取图像和JSON掩码目录
    train_img_dir = data_dict.get('train_img', data_dict.get('train', ''))
    train_mask_dir = data_dict.get('train_label', os.path.join(data_dict.get('train', ''), 'masks'))
    train_json_dir = data_dict.get('train_json', os.path.join(train_img_dir, 'json'))
    val_img_dir = data_dict.get('val_img', data_dict.get('val', ''))
    val_mask_dir = data_dict.get('val_label', os.path.join(data_dict.get('val', ''), 'masks'))
    val_json_dir = data_dict.get('val_json', os.path.join(val_img_dir, 'json'))
    num_classes = 1 if opt.single_cls else int(data_dict['nc'])
    names = data_dict['names']

    # 1. 先验证JSON文件是否存在，不存在再转换
    LOGGER.info("\n===== 开始验证JSON文件 =====")
    # 训练集JSON验证
    train_json_valid = verify_json_masks(
        img_dir=train_img_dir,
        json_mask_dir=train_json_dir
    )
    
    # 如果训练集JSON文件不完整，则进行转换
    if not train_json_valid:
        LOGGER.info("训练集JSON文件不完整，开始转换掩码为JSON...")
        train_success, train_total = batch_convert_masks_to_json(
            mask_dir=train_mask_dir,
            json_save_dir=train_json_dir
        )
        if train_success == 0 and train_total > 0:
            LOGGER.error("训练集掩码转换失败，无法继续训练")
            return (0.0,)
        
        # 重新验证
        train_json_valid = verify_json_masks(
            img_dir=train_img_dir,
            json_mask_dir=train_json_dir
        )
        if not train_json_valid:
            LOGGER.error("训练集JSON文件验证仍失败，无法继续训练")
            return (0.0,)
    
    # 验证集JSON验证
    val_json_valid = verify_json_masks(
        img_dir=val_img_dir,
        json_mask_dir=val_json_dir
    )
    
    # 如果验证集JSON文件不完整，则进行转换
    if not val_json_valid:
        LOGGER.info("验证集JSON文件不完整，开始转换掩码为JSON...")
        val_success, val_total = batch_convert_masks_to_json(
            mask_dir=val_mask_dir,
            json_save_dir=val_json_dir
        )
        if val_success == 0 and val_total > 0:
            LOGGER.error("验证集掩码转换失败，无法继续训练")
            return (0.0,)
        
        # 重新验证
        val_json_valid = verify_json_masks(
            img_dir=val_img_dir,
            json_mask_dir=val_json_dir
        )
        if not val_json_valid:
            LOGGER.error("验证集JSON文件验证仍失败，无法继续训练")
            return (0.0,)

    # 验证路径是否存在
    for path in [train_img_dir, train_json_dir, val_img_dir, val_json_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据路径不存在: {path}")
        else:
            LOGGER.info(f"找到数据路径: {path}")

    # 模型初始化
    check_suffix(weights, '.pt')
    pretrained = weights.endswith('.pt')
    model = ResNet34Seg(opt.cfg, num_classes=num_classes).to(device)
    
    # 加载预训练权重
    if pretrained:
        weights = attempt_download(weights)
        ckpt = torch.load(weights, map_location='cpu')
        exclude = []
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)
        LOGGER.info(f'加载权重: {len(csd)}/{len(model.state_dict())} 项匹配')

    # 冻结层设置
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
        batch_size = 16  # 回退到默认值

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
    best_fitness, start_epoch = 0.0, 0
    if pretrained and resume:
        best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, None, weights, epochs, resume)

    # 分布式配置
    cuda = device.type != 'cpu'
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # 数据加载器（带数据增强）
    train_loader, dataset = create_json_segment_dataloader(
        img_dir=train_img_dir,
        json_label_dir=train_json_dir,
        img_size=imgsz,
        batch_size=batch_size // WORLD_SIZE,
        augment=True,  # 训练集启用数据增强
        workers=workers,
        hyp=hyp
    )
    
    # 验证集加载器(仅主进程，不使用数据增强)
    val_loader = None
    if RANK in {-1, 0}:
        val_loader = create_json_segment_dataloader(
            img_dir=val_img_dir,
            json_label_dir=val_json_dir,
            img_size=imgsz,
            batch_size=batch_size // WORLD_SIZE * 2,
            augment=False,  # 验证集不使用数据增强
            workers=workers*2,
            shuffle=False,
            hyp=hyp
        )[0]

    # 模型属性配置
    model.num_classes = num_classes
    model.hyp = hyp
    model.class_weights = seg_labels_to_class_weights(dataset.json_files, num_classes).to(device)
    model.names = names

    # 损失函数
    criterion = SegmentationLoss(num_classes=num_classes, label_smoothing=hyp['label_smoothing'])
    amp = check_amp(model)
    scaler = torch.amp.GradScaler('cuda', enabled=amp) if amp else None

    stopper = EarlyStopping(patience=opt.patience)

    # 训练循环
    LOGGER.info(colorstr('开始训练（带数据增强的ResNet34版本）!'))
    t0 = time.time()
    global_step = 0  # 全局步数计数器，用于TensorBoard
    
    for epoch in range(start_epoch, epochs):
        model.train()
        mloss = torch.zeros(3, device=device)  # 损失缓存: 总损失, CE损失, Dice损失
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 6) % ('Epoch', 'gpu_mem', '总损失', 'CE损失', 'Dice损失', 'img_size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=len(train_loader), bar_format=TQDM_BAR_FORMAT)
        
        # 批量训练（带数据增强）
        for i, (imgs, targets, paths) in pbar:
            imgs = imgs.to(device, non_blocking=True).float() # 图像归一化
            targets = targets.to(device, non_blocking=True).long()  # 掩码是(batch_size, h, w)

            # 前向传播
            with torch.amp.autocast('cuda', enabled=amp):
                pred = model(imgs)  # 预测: (b, num_classes, h, w)
                loss, loss_items = criterion(pred, targets)  # 计算损失
                loss_items = torch.tensor(loss_items, device=device)

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
                mem = f'{torch.cuda.memory_reserved() / 1E9:.3g}G' if cuda else 'N/A'
                pbar.set_description(('%10s' * 2 + '%10.4g' * 4) %
                                    (f'{epoch}/{epochs - 1}', mem, *mloss, imgs.shape[-1]))

                # TensorBoard: 记录训练损失
                if global_step % 10 == 0:
                    tb_writer.add_scalar('Train/Total_Loss', loss_items[0], global_step)
                    tb_writer.add_scalar('Train/CE_Loss', loss_items[1], global_step)
                    tb_writer.add_scalar('Train/Dice_Loss', loss_items[2], global_step)
                    tb_writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)

                # TensorBoard: 记录训练图像示例（包括增强效果）
                if global_step % 100 == 0 and global_step != 0:
                    # 显示多个样本
                    num_samples = min(3, imgs.size(0))  # 最多显示3个样本
                    for s in range(num_samples):
                        img_np = imgs[s].permute(1, 2, 0).cpu().detach().numpy()
                        target_np = targets[s].cpu().detach().numpy()
                        pred_np = torch.argmax(pred[s], dim=0).cpu().detach().numpy()

                        target_rgb = mask_to_rgb(target_np, CAMVID_COLORS)
                        pred_rgb = mask_to_rgb(pred_np, CAMVID_COLORS)
                        diff_rgb = visualize_prediction_difference(target_np, pred_np, CAMVID_COLORS)

                        # 添加标题区分不同样本
                        tb_writer.add_image(f'Train/Sample_{s}/Input_Image', img_np, global_step, dataformats='HWC')
                        tb_writer.add_image(f'Train/Sample_{s}/Ground_Truth', target_rgb, global_step, dataformats='HWC')
                        tb_writer.add_image(f'Train/Sample_{s}/Prediction', pred_rgb, global_step, dataformats='HWC')
                        tb_writer.add_image(f'Train/Sample_{s}/Differences', diff_rgb, global_step, dataformats='HWC')

                global_step += 1

        # 学习率更新
        scheduler.step()

        # 验证
        if RANK in {-1, 0}:
            # 保存模型
            ema.update_attr(model, include=['yaml', 'num_classes', 'hyp', 'names', 'stride'])
            final_epoch = (epoch == epochs - 1)
            if not noval or final_epoch:
                # 获取验证结果
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

                # TensorBoard: 记录验证指标
                tb_writer.add_scalar('Validation/mIoU', results[0], epoch)
                for cls_idx, cls_name in enumerate(names):
                    if cls_idx < len(maps):
                        tb_writer.add_scalar(f'Validation/Class_IoU/{cls_name}', maps[cls_idx], epoch)

                # TensorBoard: 记录验证集样本可视化（重点增强第一张照片的比较）
                model.eval()
                with torch.no_grad():
                    # 获取验证集样本
                    val_batch = next(iter(val_loader))
                    val_imgs, val_targets, _ = val_batch
                    val_imgs = val_imgs.to(device)
                    val_preds = model(val_imgs)
                    val_preds = torch.argmax(val_preds, dim=1)
                    
                    # 可视化多个样本
                    num_val_samples = min(3, val_imgs.size(0))
                    for s in range(num_val_samples):
                        img_np = val_imgs[s].permute(1, 2, 0).cpu().detach().numpy()
                        target_np = val_targets[s].cpu().detach().numpy()
                        pred_np = val_preds[s].cpu().detach().numpy()
                        
                        target_rgb = mask_to_rgb(target_np, CAMVID_COLORS)
                        pred_rgb = mask_to_rgb(pred_np, CAMVID_COLORS)
                        diff_rgb = visualize_prediction_difference(target_np, pred_np, CAMVID_COLORS)
                        
                        tb_writer.add_image(f'Validation/Sample_{s}/Input_Image', img_np, epoch, dataformats='HWC')
                        tb_writer.add_image(f'Validation/Sample_{s}/Ground_Truth', target_rgb, epoch, dataformats='HWC')
                        tb_writer.add_image(f'Validation/Sample_{s}/Prediction', pred_rgb, epoch, dataformats='HWC')
                        tb_writer.add_image(f'Validation/Sample_{s}/Differences', diff_rgb, epoch, dataformats='HWC')

                    # 重点增强第一张照片的详细比较
                    if val_imgs.size(0) > 0:
                        s = 0  # 第一张照片
                        img_np = val_imgs[s].permute(1, 2, 0).cpu().detach().numpy()
                        target_np = val_targets[s].cpu().detach().numpy()
                        pred_np = val_preds[s].cpu().detach().numpy()
                        
                        target_rgb = mask_to_rgb(target_np, CAMVID_COLORS)
                        pred_rgb = mask_to_rgb(pred_np, CAMVID_COLORS)
                        diff_rgb = visualize_prediction_difference(target_np, pred_np, CAMVID_COLORS)
                        
                        # 为第一张照片创建专门的可视化组，方便对比
                        tb_writer.add_image('First_Sample/Input_Image', img_np, epoch, dataformats='HWC')
                        tb_writer.add_image('First_Sample/Ground_Truth', target_rgb, epoch, dataformats='HWC')
                        tb_writer.add_image('First_Sample/Prediction', pred_rgb, epoch, dataformats='HWC')
                        tb_writer.add_image('First_Sample/Differences', diff_rgb, epoch, dataformats='HWC')

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
                    LOGGER.info(f"更新最佳模型 (mIoU: {results[0]:.4f})")
                    
                    # 记录最佳模型预测结果 - 增强第一张照片的比较
                    with torch.no_grad():
                        model.eval()
                        sample_imgs, sample_targets, _ = next(iter(val_loader))
                        sample_imgs = sample_imgs.to(device)
                        sample_preds = model(sample_imgs)
                        sample_preds = torch.argmax(sample_preds, dim=1)
                        
                        # 重点保存第一张照片的详细对比
                        if sample_imgs.size(0) > 0:
                            s = 0  # 第一张照片
                            img_np = sample_imgs[s].permute(1, 2, 0).cpu().detach().numpy()
                            target_np = sample_targets[s].cpu().detach().numpy()
                            pred_np = sample_preds[s].cpu().detach().numpy()
                            
                            target_rgb = mask_to_rgb(target_np, CAMVID_COLORS)
                            pred_rgb = mask_to_rgb(pred_np, CAMVID_COLORS)
                            diff_rgb = visualize_prediction_difference(target_np, pred_np, CAMVID_COLORS)
                            
                            # 为最佳模型的第一张照片创建单独的TensorBoard分组
                            tb_writer.add_image('Best_Model/First_Sample/Input_Image', img_np, epoch, dataformats='HWC')
                            tb_writer.add_image('Best_Model/First_Sample/Ground_Truth', target_rgb, epoch, dataformats='HWC')
                            tb_writer.add_image('Best_Model/First_Sample/Prediction', pred_rgb, epoch, dataformats='HWC')
                            tb_writer.add_image('Best_Model/First_Sample/Differences', diff_rgb, epoch, dataformats='HWC')

            # 早停检查
            if stopper(epoch=epoch, fitness=fi):
                break

    # 训练结束
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
    # 命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='初始权重路径')
    parser.add_argument('--cfg', type=str, default='resnet34.yaml', help='模型配置文件')
    parser.add_argument('--data', type=str, default='/root/BestYOLO/CamVid/data.yaml', help='数据集配置文件（JSON版本）')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批量大小')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', default='', help='设备配置, 例如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='冻结层索引')
    parser.add_argument('--patience', type=int, default=150, help='早停 patience')
    parser.add_argument('--single-cls', action='store_true', help='单类别训练')
    parser.add_argument('--sync-bn', action='store_true', help='使用同步BN')
    parser.add_argument('--cos-lr', action='store_true', help='使用余弦学习率调度')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='恢复训练')
    parser.add_argument('--save-dir', type=str, default='runs/train-resnet34-augmented', help='结果保存目录')
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
    parser.add_argument('--hyp', type=str, default='/root/BestYOLO/data/hyps/hyp.scratch-seg.yaml', help='超参数配置文件')
    parser.add_argument('--exist-ok', action='store_true', help='允许覆盖现有实验目录')
    
    opt = parser.parse_args()
    check_requirements(exclude=('tensorboard', 'thop'))
    
    # 设备选择
    device = select_device(opt.device, batch_size=opt.batch_size)
    if device.type == 'cpu':
        opt.sync_bn = False  # CPU不支持同步BN  
    # 初始化训练
    callbacks = Callbacks()
    train(hyp=opt.hyp, opt=opt, device=device, callbacks=callbacks)
