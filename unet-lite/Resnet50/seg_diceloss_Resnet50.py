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
UTILS_ROOT = "/root/BestYOLO"
if UTILS_ROOT not in sys.path:
    sys.path.insert(0, UTILS_ROOT) 


import numpy as np
import torch
torch.use_deterministic_algorithms(False) 
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
import thop
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
# 解决PIL版本兼容性问题
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# 路径配置
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 项目根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 添加到环境变量
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 转换为相对路径

# 工具导入
import val_diceloss as validate  # 统一使用Dice损失版本的验证脚本
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


# -------------------------- 工具函数：处理YAML中的none解析 --------------------------
def parse_none(value):
    """将字符串"none"（不区分大小写）转换为Python None"""
    if isinstance(value, str) and value.lower() == 'none':
        return None
    elif isinstance(value, list):
        return [parse_none(v) for v in value]
    elif isinstance(value, dict):
        return {k: parse_none(v) for k, v in value.items()}
    return value


# -------------------------- 数据增强工具函数 --------------------------
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        return img, mask


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        return img, mask


class RandomRotation:
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
    def __init__(self, radius_range=(0.5, 2.0), p=0.2):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img, mask


class RandomCrop:
    def __init__(self, scale_range=(0.7, 1.0), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            w, h = img.size
            scale = random.uniform(*self.scale_range)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            
            x1 = random.randint(0, w - new_w)
            y1 = random.randint(0, h - new_h)
            
            img = img.crop((x1, y1, x1 + new_w, y1 + new_h))
            mask = mask.crop((x1, y1, x1 + new_w, y1 + new_h))
            
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
    try:
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask, dtype=np.int64)
        mask_np = np.clip(mask_np, 0, NUM_CLASSES - 1)
        
        json_data = {
            "filename": mask_path.name,
            "shape": mask_np.shape,
            "dtype": str(mask_np.dtype),
            "class_names": CLASS_NAMES,
            "mask_data": mask_np.flatten().tolist()
        }
        
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        LOGGER.warning(f"转换 {mask_path} 失败: {str(e)}")
        return False


def batch_convert_masks_to_json(mask_dir, json_save_dir):
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
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    img_files = [f for f in Path(img_dir).iterdir() if f.suffix.lower() in img_extensions]
    
    if not img_files:
        LOGGER.warning(f"在 {img_dir} 中未找到任何图像文件")
        return False
    
    missing = []
    for img_file in img_files:
        json_file = Path(json_mask_dir) / f"{img_file.stem}.json"
        if not json_file.exists():
            missing.append(img_file.name)
    
    if missing:
        LOGGER.warning(f"发现 {len(missing)} 个图像缺少对应的JSON掩码:")
        for name in missing[:10]:
            LOGGER.warning(f"  - {name}")
        if len(missing) > 10:
            LOGGER.warning(f"  ... 还有 {len(missing)-10} 个缺失文件")
        return False
    else:
        LOGGER.info(f"所有 {len(img_files)} 个图像都有对应的JSON掩码")
        return True


# -------------------------- 语义分割数据集类（JSON版本） --------------------------
class JSONSegmentDataset(Dataset):
    def __init__(self, img_dir, json_label_dir, img_size=640, augment=False, hyp=None):
        self.img_dir = Path(img_dir)
        self.json_label_dir = Path(json_label_dir)
        self.img_size = img_size
        self.augment = augment
        self.augmentations = get_augmentations(hyp) if (augment and hyp is not None) else []
        
        self.img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.img_files = [f for f in self.img_dir.iterdir() if f.suffix.lower() in self.img_extensions]
        
        if not self.img_files:
            raise FileNotFoundError(f"在 {img_dir} 中未找到任何图像文件")
        
        self.json_files = []
        for img_file in self.img_files:
            json_file = self.json_label_dir / f"{img_file.stem}.json"
            if not json_file.exists():
                raise FileNotFoundError(f"JSON掩码文件不存在: {json_file}")
            self.json_files.append(json_file)
        
        with open(self.json_files[0], "r") as f:
            first_json = json.load(f)
        self.class_names = first_json["class_names"]
        self.num_classes = len(self.class_names)
        
        LOGGER.info(f"加载JSON分割数据集: {len(self.img_files)} 张图像，{len(self.json_files)} 个JSON掩码，数据增强: {augment}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        json_path = self.json_files[index]
        
        img = Image.open(img_path).convert('RGB')
        with open(json_path, "r") as f:
            json_data = json.load(f)
        
        mask_shape = json_data["shape"]
        mask_flat = np.array(json_data["mask_data"], dtype=np.int64)
        mask_np = mask_flat.reshape(mask_shape)
        mask_np = np.clip(mask_np, 0, self.num_classes - 1)
        mask = Image.fromarray(mask_np.astype(np.uint8))
        
        if self.augment and self.augmentations:
            img, mask = self._apply_augmentations(img, mask)
        
        img, mask = self._resize_and_pad(img, mask)
        
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).long()
        
        return img, mask, str(img_path)

    def _apply_augmentations(self, img, mask):
        augmented = random.sample(self.augmentations, k=len(self.augmentations))
        for aug in augmented:
            img, mask = aug(img, mask)
        return img, mask

    def _resize_and_pad(self, img, mask):
        w, h = img.size
        scale = min(self.img_size / w, self.img_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        
        pad_w = self.img_size - new_w
        pad_h = self.img_size - new_h
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        
        new_img = Image.new('RGB', (self.img_size, self.img_size), (128, 128, 128))
        new_label = Image.new('L', (self.img_size, self.img_size), 0)
        
        new_img.paste(img, (pad_left, pad_top))
        new_label.paste(mask, (pad_left, pad_top))
        
        return new_img, new_label


# -------------------------- 语义分割数据加载器 --------------------------
def create_json_segment_dataloader(img_dir, json_label_dir, img_size=640, batch_size=16, 
                                  augment=False, workers=8, shuffle=True, hyp=None):
    dataset = JSONSegmentDataset(
        img_dir=img_dir,
        json_label_dir=json_label_dir,
        img_size=img_size,
        augment=augment,
        hyp=hyp
    )
    
    if batch_size > len(dataset):
        batch_size = len(dataset)
        LOGGER.warning(f"批次大小调整为 {batch_size}（小于请求的批次，因数据集过小）")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader, dataset


# -------------------------- YAML中所有模块的核心实现 --------------------------
def autopad(k: int, p: Optional[int] = None) -> int:
    """自动计算卷积填充（适配所有卷积模块）"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """标准卷积模块（YAML中Conv模块的实现）：Conv2d -> BatchNorm2d -> ReLU"""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None, 
                 g: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class BottleneckBlock(nn.Module):
    """ResNet50瓶颈块（ResNet50Layer的内部单元），通道扩展倍数为4"""
    expansion = 4  # 输出通道 = 中间通道 × expansion

    def __init__(self, in_channels: int, mid_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        # 1x1卷积降维
        self.conv1 = Conv(in_channels, mid_channels, k=1, s=1, p=0, act=True)
        # 3x3卷积提取特征
        self.conv2 = Conv(mid_channels, mid_channels, k=3, s=stride, p=1, act=True)
        # 1x1卷积升维（扩展4倍）
        self.conv3 = Conv(mid_channels, mid_channels * self.expansion, k=1, s=1, p=0, act=False)
        # 下采样模块（匹配通道数和步长）
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x  # 残差连接
        
        # 瓶颈块前向
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        # 下采样（若需要）
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 残差相加 + 激活
        out += identity
        return self.act(out)


class ResNetStem(nn.Module):
    """ResNet初始模块（YAML中ResNetStem的实现）：7x7卷积 + 3x3池化"""
    def __init__(self, out_channels: int):
        super().__init__()
        self.stem = nn.Sequential(
            Conv(3, out_channels, k=7, s=2, p=3),  # 7x7卷积（步长2，输出通道由args[0]指定）
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 3x3池化（步长2，尺寸再减半）
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class ResNet50Layer(nn.Module):
    """ResNet50层（YAML中ResNet50Layer的实现）：多个BottleneckBlock的序列"""
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 1):
        super().__init__()
        mid_channels = out_channels // BottleneckBlock.expansion  # 中间通道 = 输出通道 / 4（因expansion=4）
        self.downsample = None
        
        # 第一个瓶颈块：若步长≠1或输入输出通道不匹配，需要下采样
        if stride != 1 or in_channels != out_channels:
            self.downsample = Conv(in_channels, out_channels, k=1, s=stride, p=0, act=False)
        
        # 构建瓶颈块序列
        blocks = [BottleneckBlock(in_channels, mid_channels, stride, self.downsample)]
        for _ in range(1, num_blocks):
            blocks.append(BottleneckBlock(out_channels, mid_channels))  # 后续块输入通道=输出通道
        
        self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class SPPF(nn.Module):
    """空间金字塔池化-快速版（YAML中SPPF的实现）：增强感受野"""
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2  # 隐藏层通道数（输入通道的1/2）
        self.cv1 = Conv(c1, c_, 1, 1)  # 1x1卷积降维
        self.cv2 = Conv(c_ * 4, c2, 1, 1)  # 1x1卷积融合4个分支特征
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 3次最大池化（同核大小）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)    # 第一次池化
        y2 = self.m(y1)   # 第二次池化
        y3 = self.m(y2)   # 第三次池化
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))  # 拼接4个特征图


class Upsample(nn.Module):
    """上采样模块（YAML中Upsample的实现）：支持nearest/bilinear等模式"""
    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super().__init__()
        self.upsample = nn.Upsample(
            size=size, 
            scale_factor=scale_factor, 
            mode=mode, 
            align_corners=False if mode != 'nearest' else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class Concat(nn.Module):
    """特征拼接模块（YAML中Concat的实现）：按通道维度拼接"""
    def __init__(self, dimension: int = 1):
        super().__init__()
        self.dim = dimension  # 拼接维度（默认通道维度1）

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        # 对齐所有输入特征的尺寸（以第一个特征图尺寸为基准）
        target_size = x[0].shape[2:]
        aligned = []
        for tensor in x:
            if tensor.shape[2:] != target_size:
                tensor = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
            aligned.append(tensor)
        return torch.cat(aligned, dim=self.dim)


class C3(nn.Module):
    """特征融合模块（YAML中C3的实现）：用于分割头特征提取"""
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)  # 分支通道数（输出通道的1/2）
        self.cv1 = Conv(c1, self.c, 1, 1)  # 分支1：1x1降维 + 3x3卷积序列
        self.cv2 = Conv(c1, self.c, 1, 1)  # 分支2：直接1x1降维
        self.cv3 = Conv(2 * self.c, c2, 1, 1)  # 拼接后1x1升维
        # 3x3卷积序列（n个Conv模块）
        self.m = nn.Sequential(*(Conv(self.c, self.c, 3, 1, g=g) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 分支1：降维 → 多次卷积；分支2：直接降维；拼接 → 升维
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


# -------------------------- ResNet50Seg 语义分割模型（解析YAML所有模块） --------------------------
class ResNet50Seg(nn.Module):
    """ResNet50语义分割模型：编码器（ResNet50）+ 解码器（YAML定义的分割头）"""
    def __init__(self, cfg: Dict, num_classes: Optional[int] = None):
        super().__init__()
        # 解析配置文件
        if isinstance(cfg, str):
            cfg = check_yaml(cfg)
            with open(cfg, 'r') as f:
                raw_yaml = yaml.safe_load(f)
            # 核心修复：处理YAML中的"none"字符串，转换为Python None
            self.yaml = parse_none(raw_yaml)
        else:
            self.yaml = parse_none(cfg)
        
        # 配置参数
        self.num_classes = self.yaml['nc'] if num_classes is None else num_classes
        self.yaml['nc'] = self.num_classes
        self.img_size = [640, 640]
        self.stride = torch.tensor([4, 8, 16, 32])  # ResNet50各阶段下采样倍数
        
        # 添加names属性，用于验证脚本
        self.names = CLASS_NAMES  # 使用全局定义的类别名称列表
        
        # 构建主干网络和分割头（核心：解析YAML中的所有模块）
        self.backbone, self.backbone_out_chs = self._build_backbone(self.yaml['backbone'])
        self.head, self.head_out_chs = self._build_head(self.yaml['head'], self.backbone_out_chs)
        
        # 初始化权重与日志
        self._initialize_weights()
        self._log_model_info()

    def _build_backbone(self, cfg):
        """构建Backbone（解析YAML中的ResNetStem、ResNet50Layer）"""
        backbone = nn.ModuleList()
        out_chs = []
        prev_out_ch = 3  # 初始输入通道（RGB）

        for layer_idx, layer in enumerate(cfg):
            from_, num_, module, args = layer
            # 计算输入通道（from=-1表示上一层，否则取指定层的输出通道）
            c1 = prev_out_ch if from_ == -1 else out_chs[from_]

            # 解析Backbone模块（YAML中定义的ResNetStem、ResNet50Layer）
            if module == 'ResNetStem':
                out_ch = int(args[0])  # 确保输出通道为整数
                layer_module = ResNetStem(out_channels=out_ch)
            elif module == 'ResNet50Layer':
                out_ch = int(args[0])  # 输出通道（如256、512）
                num_blocks = int(args[1])  # 瓶颈块数量（如3、4）
                stride = int(args[2]) if len(args) >= 3 else 1  # 步长（如1、2）
                layer_module = ResNet50Layer(in_channels=c1, out_channels=out_ch, num_blocks=num_blocks, stride=stride)
            else:
                raise NotImplementedError(f"Backbone未知模块: {module}")

            backbone.append(layer_module)
            out_chs.append(out_ch)
            prev_out_ch = out_ch
            LOGGER.debug(f"Backbone层 {layer_idx}: {module} → 输入{c1}→输出{out_ch}通道")

        return backbone, out_chs

    def _build_head(self, cfg, backbone_out_chs):
        """构建分割头（解析YAML中的Conv、SPPF、Upsample、Concat、C3、nn.Softmax）"""
        head = nn.ModuleList()
        all_out_chs = backbone_out_chs.copy()  # 记录所有层的输出通道（含Backbone和Head）

        for layer_idx, layer in enumerate(cfg):
            from_, num_, module, args = layer
            # 1. 计算输入通道（支持多输入from，如[[-1,8]]表示取上一层和第8层）
            if isinstance(from_, list):
                c1 = sum([all_out_chs[f] for f in from_])  # 多输入时通道求和
            else:
                c1 = all_out_chs[from_]  # 单输入时取指定层通道

            # 2. 解析Head模块（YAML中定义的所有模块）
            if module == 'Conv':
                # args格式：[out_ch, k, s, p, g, act]（后续参数可选）
                # 强制转换所有数值参数为整数
                out_ch = int(args[0])  # 输出通道必须是整数
                k = int(args[1]) if len(args) >= 2 else 1  # 卷积核大小
                s = int(args[2]) if len(args) >= 3 else 1  # 步长
                p = int(args[3]) if (len(args) >= 4 and args[3] is not None) else None  # padding
                g = int(args[4]) if len(args) >= 5 else 1  # groups（强制转为整数）
                act = args[5] if len(args) >= 6 else True  # 激活函数（布尔值）
                layer_module = Conv(c1, out_ch, k, s, p, g, act)

            elif module == 'SPPF':
                # args格式：[out_ch, k]（k可选，默认5）
                out_ch = int(args[0])
                k = int(args[1]) if len(args) >= 2 else 5
                layer_module = SPPF(c1, out_ch, k)

            elif module == 'Upsample':
                # args格式：[size, scale_factor, mode]
                # 核心修复：处理none并确保参数不冲突
                size = args[0] if len(args) >= 1 else None
                scale_factor = args[1] if len(args) >= 2 else 2.0
                mode = args[2] if len(args) >= 3 else 'nearest'
                
                # 转换数值类型
                if scale_factor is not None:
                    scale_factor = float(scale_factor)
                # 确保size和scale_factor不同时生效
                if size is not None and scale_factor is not None:
                    LOGGER.warning(f"Upsample层{layer_idx}：同时设置size和scale_factor，自动使用scale_factor={scale_factor}")
                    size = None  # 优先使用scale_factor
                
                layer_module = Upsample(size=size, scale_factor=scale_factor, mode=mode)
                out_ch = c1  # 上采样不改变通道数

            elif module == 'Concat':
                # args格式：[dimension]（默认通道维度1）
                dim = int(args[0]) if len(args) >= 1 else 1
                layer_module = Concat(dimension=dim)
                out_ch = c1  # 拼接前已计算通道和，此处直接沿用c1

            elif module == 'C3':
                # args格式：[out_ch, n, shortcut, g, e]（后续参数可选）
                out_ch = int(args[0])
                n = int(args[1]) if len(args) >= 2 else 1
                shortcut = bool(args[2]) if len(args) >= 3 else False
                g = int(args[3]) if len(args) >= 4 else 1
                e = float(args[4]) if len(args) >= 5 else 0.5
                layer_module = C3(c1, out_ch, n, shortcut, g, e)

            elif module == 'nn.Softmax':
                # args格式：[dim]（默认通道维度1）
                dim = int(args[0]) if len(args) >= 1 else 1
                layer_module = nn.Softmax(dim=dim)
                out_ch = c1  # Softmax不改变通道数

            else:
                raise NotImplementedError(f"分割头未知模块: {module}")

            # 3. 记录当前层信息
            head.append(layer_module)
            all_out_chs.append(out_ch)
            LOGGER.debug(f"Head层 {layer_idx}: {module} → 输入{c1}→输出{out_ch}通道")

        return head, all_out_chs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：Backbone → Head"""
        # 1. Backbone前向（保存所有层输出，供Head调用）
        x_outs = []
        for layer in self.backbone:
            x = layer(x)
            x_outs.append(x)
        
        # 2. Head前向（按YAML配置调用对应层的输入）
        for layer, layer_cfg in zip(self.head, self.yaml['head']):
            from_, num_, module, args = layer_cfg
            # 获取输入特征（支持单输入/多输入）
            if isinstance(from_, list):
                inputs = [x_outs[f] for f in from_]
            else:
                inputs = x_outs[from_]
            
            # 执行模块计算
            if isinstance(layer, Concat):
                x = layer(inputs)  # Concat需要多输入列表
            else:
                x = layer(inputs)  # 其他模块单输入张量
            
            # 保存当前层输出，供后续层调用
            x_outs.append(x)
        
        # 确保输出尺寸与输入一致（防止上采样误差）
        if x.shape[2:] != (self.img_size[0], self.img_size[1]):
            x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        
        return x

    def _initialize_weights(self) -> None:
        """ResNet风格权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _log_model_info(self) -> None:
        """打印模型信息"""
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        input_tensor = torch.randn(1, 3, *self.img_size)
        try:
            # 处理可能的thop计算错误
            flops, _ = thop.profile(deepcopy(self), inputs=(input_tensor,), verbose=False)
            flops_info = f"{flops / 1e9:.2f} GFLOPs"
        except Exception as e:
            flops_info = f"无法计算 (错误: {str(e)})"
        
        LOGGER.info(f"ResNet50Seg模型信息: {n_params:,} 总参数, {n_trainable:,} 可训练参数")
        LOGGER.info(f"计算量: {flops_info} (输入尺寸: {self.img_size})")
        LOGGER.info(f"类别数: {self.num_classes}")


# -------------------------- 语义分割损失函数（Dice+交叉熵） --------------------------
class SegmentationLoss(nn.Module):
    """语义分割损失：交叉熵损失 + 带权重Dice损失"""
    def __init__(self, num_classes: int = 12, label_smoothing: float = 0.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
        # 类别权重（默认全1）
        if class_weights is not None:
            self.class_weights = class_weights if isinstance(class_weights, torch.Tensor) else torch.tensor(class_weights)
        else:
            self.class_weights = torch.ones(num_classes)
            
        # 带权重的交叉熵损失
        self.cross_entropy = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=label_smoothing)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        # 权重设备对齐
        if self.class_weights.device != pred.device:
            self.class_weights = self.class_weights.to(pred.device)
            self.cross_entropy.weight = nn.Parameter(self.class_weights)
            
        # 尺寸对齐（若pred与target尺寸不一致，上采样target）
        if pred.shape[2:] != target.shape[1:]:
            target = F.interpolate(target.unsqueeze(1).float(), size=pred.shape[2:], mode='nearest').squeeze(1).long()
        
        # 计算损失
        ce_loss = self.cross_entropy(pred, target)
        dice_loss = self._dice_loss(pred.softmax(1), self._one_hot_encode(target))
        total_loss = ce_loss + 0.5 * dice_loss  # 损失加权
        
        return total_loss, [total_loss.item(), ce_loss.item(), dice_loss.item()]

    def _one_hot_encode(self, target: torch.Tensor) -> torch.Tensor:
        """标签转为One-Hot格式"""
        b, h, w = target.shape
        one_hot = torch.zeros(b, self.num_classes, h, w, device=target.device)
        return one_hot.scatter_(1, target.unsqueeze(1), 1.0)

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """带类别权重的Dice损失"""
        weighted_pred = pred * self.class_weights.view(1, -1, 1, 1).to(pred.device)
        intersection = (weighted_pred * target).sum(dim=(2, 3))
        pred_sum = weighted_pred.sum(dim=(2, 3))
        target_sum = target.sum(dim=(2, 3))
        dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
        return 1.0 - dice.mean()


# -------------------------- 类别权重计算与解析 --------------------------
def seg_labels_to_class_weights(json_files, num_classes):
    """从JSON掩码计算类别权重（数据驱动）"""
    class_counts = np.zeros(num_classes, dtype=np.int64)
    total = 0
    
    for file in json_files:
        try:
            with open(file, "r") as f:
                json_data = json.load(f)
            mask_flat = np.array(json_data["mask_data"], dtype=np.int64)
            counts = np.bincount(mask_flat, minlength=num_classes)
            class_counts += counts
            total += len(mask_flat)
        except Exception as e:
            LOGGER.warning(f"处理JSON文件 {file} 时出错: {str(e)}")
            continue
    
    weights = total / (num_classes * (class_counts + 1e-8))  # 避免除以0
    return torch.from_numpy(weights).float()


def parse_class_weights(weight_str, num_classes):
    """解析自定义类别权重（文件或字符串）"""
    if not weight_str:
        return None
        
    # 从文件加载（yaml格式）
    if os.path.exists(weight_str):
        try:
            with open(weight_str, 'r') as f:
                weights = yaml.safe_load(f)
            # 处理可能的"none"值
            weights = parse_none(weights)
            if isinstance(weights, dict):
                # 字典格式：{class_name: weight}
                weight_list = [weights[cls] for cls in CLASS_NAMES[:num_classes]]
            elif isinstance(weights, list):
                # 列表格式：[w0, w1, ..., w11]
                if len(weights) != num_classes:
                    raise ValueError(f"权重数量与类别数不匹配: {len(weights)} vs {num_classes}")
                weight_list = weights
            else:
                raise ValueError("权重文件格式必须为字典或列表")
            return torch.tensor(weight_list, dtype=torch.float32)
        except Exception as e:
            LOGGER.error(f"加载权重文件失败: {e}")
            raise
    
    # 从字符串解析（如"1.0,2.0,...,1.0"）
    try:
        weight_list = list(map(float, weight_str.split(',')))
        if len(weight_list) != num_classes:
            raise ValueError(f"权重数量与类别数不匹配: {len(weight_list)} vs {num_classes}")
        return torch.tensor(weight_list, dtype=torch.float32)
    except Exception as e:
        LOGGER.error(f"解析权重字符串失败: {e}")
        raise


# -------------------------- 可视化工具函数 --------------------------
def mask_to_rgb(mask, color_map):
    """掩码转为RGB图像（用于可视化）"""
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(len(color_map)):
        rgb_mask[mask == cls] = color_map[cls]
    return rgb_mask


def visualize_prediction_difference(gt_mask, pred_mask, color_map):
    """可视化预测与真值的差异（正确：原颜色，错误：洋红色）"""
    h, w = gt_mask.shape
    diff_mask = np.zeros((h, w, 3), dtype=np.uint8)
    # 错误区域：洋红色
    diff_regions = gt_mask != pred_mask
    diff_mask[diff_regions] = [255, 0, 255]
    # 正确区域：原类别颜色
    for cls in range(len(color_map)):
        correct_regions = (gt_mask == cls) & (pred_mask == cls)
        diff_mask[correct_regions] = color_map[cls]
    return diff_mask


# -------------------------- 训练主函数 --------------------------
def train(hyp: Dict, opt: argparse.Namespace, device: torch.device, callbacks: Callbacks) -> Tuple[float, ...]:
    # 基础配置
    save_dir = Path(opt.save_dir)
    epochs, batch_size, weights = opt.epochs, opt.batch_size, opt.weights
    resume, noval, nosave = opt.resume, opt.noval, opt.nosave
    workers, freeze = opt.workers, opt.freeze
    
    # 创建保存目录
    w = save_dir / 'weights'
    w.mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'

    # 处理超参数（添加文件不存在的容错逻辑，使用内置默认超参数）
    default_hyp = {
        'lr0': 0.01, 'lrf': 0.2, 'momentum': 0.937, 'weight_decay': 0.0005,
        'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1,
        'label_smoothing': 0.0, 'degrees': 10.0, 'flipud': 0.2, 'fliplr': 0.5
    }
    
    if isinstance(hyp, str):
        try:
            # 尝试加载超参数文件，失败则用默认值
            with open(hyp) as f:
                hyp = yaml.safe_load(f)
            # 处理hyp中的none值
            hyp = parse_none(hyp)
            LOGGER.info(f"成功加载超参数文件: {hyp}")
        except FileNotFoundError:
            LOGGER.warning(f"未找到超参数文件 {hyp}，自动使用内置默认超参数")
            hyp = default_hyp.copy()
    else:
        hyp = default_hyp.copy()
    
    # 补充缺失的超参数
    for k, v in default_hyp.items():
        if k not in hyp:
            hyp[k] = v
            LOGGER.info(f"补充超参数: {k} = {v}")
    yaml_save(save_dir / 'hyp.yaml', hyp)
    yaml_save(save_dir / 'opt.yaml', vars(opt))

    # 日志初始化
    data_dict = None
    loggers = None
    tb_writer = None
    if RANK in {-1, 0}:
        experiment_dir = increment_path(save_dir / "exp", exist_ok=opt.exist_ok)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=experiment_dir)
        # 修复noplots参数：若opt无noplots属性，自动添加并设为False
        if not hasattr(opt, 'noplots'):
            opt.noplots = False
        # 确保evolve属性存在
        if not hasattr(opt, 'evolve'):
            opt.evolve = False
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
        if not isinstance(callbacks, Callbacks):
            callbacks = Callbacks()
        # 注册日志回调
        for k in dir(loggers):
            if k.startswith('on_') and callable(getattr(loggers, k)):
                try:
                    callbacks.register_action(k, getattr(loggers, k))
                except Exception as e:
                    LOGGER.warning(f"注册回调 {k} 失败: {e}")
        data_dict = loggers.remote_dataset

    # 数据集配置与验证
    init_seeds(seed=(opt.seed + RANK) % (2**32), deterministic=False)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(opt.data)
    # 数据路径解析
    train_img_dir = data_dict.get('train_img', data_dict.get('train', ''))
    train_mask_dir = data_dict.get('train_label', os.path.join(train_img_dir, 'masks'))
    train_json_dir = data_dict.get('train_json', os.path.join(train_img_dir, 'json'))
    val_img_dir = data_dict.get('val_img', data_dict.get('val', ''))
    val_mask_dir = data_dict.get('val_label', os.path.join(val_img_dir, 'masks'))
    val_json_dir = data_dict.get('val_json', os.path.join(val_img_dir, 'json'))
    num_classes = 1 if opt.single_cls else int(data_dict['nc'])
    names = data_dict['names']

    # 验证/转换JSON掩码（确保数据完整性）
    LOGGER.info("\n===== 验证JSON掩码 =====")
    # 训练集
    if not verify_json_masks(train_img_dir, train_json_dir):
        LOGGER.info("训练集JSON不完整，开始转换掩码→JSON...")
        batch_convert_masks_to_json(train_mask_dir, train_json_dir)
        if not verify_json_masks(train_img_dir, train_json_dir):
            LOGGER.error("训练集JSON验证失败，终止训练")
            return (0.0,)
    # 验证集
    if not verify_json_masks(val_img_dir, val_json_dir):
        LOGGER.info("验证集JSON不完整，开始转换掩码→JSON...")
        batch_convert_masks_to_json(val_mask_dir, val_json_dir)
        if not verify_json_masks(val_img_dir, val_json_dir):
            LOGGER.error("验证集JSON验证失败，终止训练")
            return (0.0,)

    # 模型初始化（加载配置+权重）
    check_suffix(weights, '.pt')
    pretrained = weights.endswith('.pt')
    model = ResNet50Seg(opt.cfg, num_classes=num_classes).to(device)  # 使用自定义ResNet50Seg模型
    # 加载预训练权重
    if pretrained:
        weights = attempt_download(weights)
        ckpt = torch.load(weights, map_location='cpu')
        csd = ckpt['model'].float().state_dict() if 'model' in ckpt else ckpt.float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=[])  # 只加载匹配的权重
        model.load_state_dict(csd, strict=False)
        LOGGER.info(f"加载权重: {len(csd)}/{len(model.state_dict())} 项匹配")

    # 冻结层配置
    freeze_layers = [f'backbone.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = not any(x in k for x in freeze_layers)
        if not v.requires_grad:
            LOGGER.info(f"冻结层: {k}")

    # 图像尺寸检查（确保符合下采样倍数）
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    # 优化器与学习率调度
    nbs = 64  # 名义批次大小
    accumulate = max(round(nbs / batch_size), 1)  # 梯度累积次数
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # 调整权重衰减
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    # 余弦学习率/线性学习率
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA（指数移动平均）与恢复训练
    best_fitness, start_epoch = 0.0, 0
    if pretrained and resume:
        best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, None, weights, epochs, resume)

    # 分布式配置
    cuda = device.type != 'cpu'
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)  # 单机多卡
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)  # 同步BN
    ema = ModelEMA(model) if RANK in {-1, 0} else None  # 仅主进程使用EMA

    # 数据加载器
    train_loader, dataset = create_json_segment_dataloader(
        img_dir=train_img_dir,
        json_label_dir=train_json_dir,
        img_size=imgsz,
        batch_size=batch_size // WORLD_SIZE,
        augment=True,
        workers=workers,
        hyp=hyp
    )
    val_loader = create_json_segment_dataloader(
        img_dir=val_img_dir,
        json_label_dir=val_json_dir,
        img_size=imgsz,
        batch_size=batch_size // WORLD_SIZE * 2,
        augment=False,
        workers=workers*2,
        shuffle=False,
        hyp=hyp
    )[0] if RANK in {-1, 0} else None

    # 类别权重配置
    if opt.class_weights:
        class_weights = parse_class_weights(opt.class_weights, num_classes)
    else:
        class_weights = seg_labels_to_class_weights(dataset.json_files, num_classes)
    class_weights = class_weights.to(device)
    LOGGER.info(f"类别权重: {class_weights.tolist()}")

    # 损失函数（带类别权重的Dice+交叉熵）
    criterion = SegmentationLoss(
        num_classes=num_classes,
        label_smoothing=hyp['label_smoothing'],
        class_weights=class_weights
    )

    # AMP（自动混合精度）配置
    amp_enabled = opt.amp and cuda
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None
    LOGGER.info(f"AMP训练: {'启用' if amp_enabled else '禁用'}")

    # 早停配置
    stopper = EarlyStopping(patience=opt.patience)

    # 训练循环
    LOGGER.info(colorstr('开始训练（ResNet50Seg + Dice损失）!'))
    t0 = time.time()
    global_step = 0
    
    for epoch in range(start_epoch, epochs):
        model.train()
        mloss = torch.zeros(3, device=device)  # 总损失、CE损失、Dice损失
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 6) % ('Epoch', 'GPU内存', '总损失', 'CE损失', 'Dice损失', '图像尺寸'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=len(train_loader), bar_format=TQDM_BAR_FORMAT)
        
        for i, (imgs, targets, paths) in pbar:
            # 数据设备转移
            imgs = imgs.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).long()

            # 前向传播（AMP加速）
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred = model(imgs)
                loss, loss_items = criterion(pred, targets)  # loss_items: [总损失, CE, Dice]

            # 反向传播与梯度累积
            loss /= accumulate  # 梯度累积：损失除以累积次数
            if amp_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度更新（每accumulate次或最后一次迭代）
            if (i + 1) % accumulate == 0 or i == len(train_loader) - 1:
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)  # 更新EMA模型

            # 日志更新
            if RANK in {-1, 0}:
                mloss = (mloss * i + torch.tensor(loss_items, device=device)) / (i + 1)  # 移动平均损失
                mem = f'{torch.cuda.memory_reserved() / 1E9:.3g}G' if cuda else 'N/A'
                pbar.set_description(('%10s' * 2 + '%10.4g' * 4) %
                                    (f'{epoch}/{epochs - 1}', mem, *mloss, imgs.shape[-1]))

                # TensorBoard日志（每10步记录损失，每100步记录可视化）
                if global_step % 10 == 0:
                    tb_writer.add_scalar('Train/Total_Loss', loss_items[0], global_step)
                    tb_writer.add_scalar('Train/CE_Loss', loss_items[1], global_step)
                    tb_writer.add_scalar('Train/Dice_Loss', loss_items[2], global_step)
                    tb_writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)

                if global_step % 100 == 0:
                    # 可视化训练样本（输入、真值、预测、差异）
                    num_samples = min(3, imgs.size(0))
                    for s in range(num_samples):
                        img_np = imgs[s].permute(1, 2, 0).cpu().detach().numpy()
                        target_np = targets[s].cpu().detach().numpy()
                        pred_np = torch.argmax(pred[s], dim=0).cpu().detach().numpy()
                        # 转为RGB可视化
                        target_rgb = mask_to_rgb(target_np, CAMVID_COLORS)
                        pred_rgb = mask_to_rgb(pred_np, CAMVID_COLORS)
                        diff_rgb = visualize_prediction_difference(target_np, pred_np, CAMVID_COLORS)
                        # 写入TensorBoard
                        tb_writer.add_image(f'Train/Sample_{s}/Input', img_np, global_step, dataformats='HWC')
                        tb_writer.add_image(f'Train/Sample_{s}/GT', target_rgb, global_step, dataformats='HWC')
                        tb_writer.add_image(f'Train/Sample_{s}/Pred', pred_rgb, global_step, dataformats='HWC')
                        tb_writer.add_image(f'Train/Sample_{s}/Diff', diff_rgb, global_step, dataformats='HWC')

                global_step += 1

        # 学习率更新
        scheduler.step()

        # 验证与模型保存（仅主进程）
        if RANK in {-1, 0}:
            ema.update_attr(model, include=['yaml', 'num_classes', 'hyp', 'names', 'stride'])
            final_epoch = epoch == epochs - 1
            if not noval or final_epoch:
                # 验证（使用EMA模型，更稳定）
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
                # 验证日志
                tb_writer.add_scalar('Val/mIoU', results[0], epoch)
                for cls_idx, cls_name in enumerate(names):
                    if cls_idx < len(maps):
                        tb_writer.add_scalar(f'Val/IoU_{cls_name}', maps[cls_idx], epoch)

                # 验证集可视化
                model.eval()
                with torch.no_grad():
                    val_imgs, val_targets, _ = next(iter(val_loader))
                    val_imgs = val_imgs.to(device)
                    val_preds = model(val_imgs)
                    val_preds = torch.argmax(val_preds, dim=1)
                    # 可视化3个样本
                    num_val_samples = min(3, val_imgs.size(0))
                    for s in range(num_val_samples):
                        img_np = val_imgs[s].permute(1, 2, 0).cpu().detach().numpy()
                        target_np = val_targets[s].cpu().detach().numpy()
                        pred_np = val_preds[s].cpu().detach().numpy()
                        # 转为RGB
                        target_rgb = mask_to_rgb(target_np, CAMVID_COLORS)
                        pred_rgb = mask_to_rgb(pred_np, CAMVID_COLORS)
                        diff_rgb = visualize_prediction_difference(target_np, pred_np, CAMVID_COLORS)
                        # 写入TensorBoard
                        tb_writer.add_image(f'Val/Sample_{s}/Input', img_np, epoch, dataformats='HWC')
                        tb_writer.add_image(f'Val/Sample_{s}/GT', target_rgb, epoch, dataformats='HWC')
                        tb_writer.add_image(f'Val/Sample_{s}/Pred', pred_rgb, epoch, dataformats='HWC')
                        tb_writer.add_image(f'Val/Sample_{s}/Diff', diff_rgb, epoch, dataformats='HWC')

            # 模型保存（last.pt保存最新，best.pt保存最优）
            fi = fitness(np.array(results).reshape(1, -1))  # 适应度（mIoU越高越好）
            if fi > best_fitness:
                best_fitness = fi
            if not nosave or final_epoch:
                torch.save({
                    'model': ema.ema,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_fitness': best_fitness
                }, last)
                if fi == best_fitness:
                    torch.save({'model': ema.ema}, best)
                    LOGGER.info(f"更新最佳模型（mIoU: {results[0]:.4f}）")

            # 早停检查
            if stopper(epoch=epoch, fitness=fi):
                LOGGER.info(f"早停触发（epoch {epoch}）")
                break

    # 训练结束
    if RANK in {-1, 0}:
        tb_writer.close()
        LOGGER.info(f'\n训练完成（耗时: {(time.time() - t0) / 3600:.2f} 小时）')
        LOGGER.info(f"最佳模型路径: {best}")
        strip_optimizer(best)  # 移除优化器状态，减小模型体积

    return best_fitness


# -------------------------- 命令行参数与主函数 --------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='预训练权重路径（可选）')
    parser.add_argument('--cfg', type=str, default='resnet50.yaml', help='模型配置文件（必填，即你的YAML路径）')
    parser.add_argument('--data', type=str, default='/root/BestYOLO/CamVid/data.yaml', help='数据集配置文件')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4, help='批量大小')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', default='', help='设备（0/0,1/cpu）')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='冻结Backbone前N层(如[2]冻结前2层)')
    parser.add_argument('--patience', type=int, default=150, help='早停耐心值')
    parser.add_argument('--single-cls', action='store_true', help='单类别训练（CamVid无需启用）')
    parser.add_argument('--sync-bn', action='store_true', help='分布式训练同步BN（多卡启用）')
    parser.add_argument('--cos-lr', action='store_true', help='余弦学习率调度')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='恢复训练')
    parser.add_argument('--save-dir', type=str, default='runs/train-resnet50-seg', help='结果保存目录')
    parser.add_argument('--optimizer', type=str, default='SGD', help='优化器（SGD/Adam）')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='标签平滑因子')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--noval', action='store_true', help='禁用验证（不推荐）')
    parser.add_argument('--nosave', action='store_true', help='不保存模型（不推荐）')
    parser.add_argument('--exist-ok', action='store_true', help='允许覆盖现有实验目录')
    parser.add_argument('--amp', action='store_true', default=False, help='启用AMP混合精度训练')
    parser.add_argument('--class-weights', type=str, default='weight.yaml', help='自定义类别权重（文件/字符串，可选）')
    parser.add_argument('--hyp', type=str, default='/root/BestYOLO/data/hyps/hyp.scratch-seg.yaml', help='超参数配置文件路径（可选）')
    parser.add_argument('--evolve', action='store_true', help='启用超参数进化搜索（默认禁用）')
    parser.add_argument('--noplots', action='store_true', help='禁用结果绘图（修复Loggers依赖）')
    opt = parser.parse_args()
    check_requirements(exclude=('tensorboard', 'thop'))  # 检查依赖
    
    # 设备选择
    device = select_device(opt.device, batch_size=opt.batch_size)
    if device.type == 'cpu':
        opt.sync_bn = False  # CPU不支持同步BN
        opt.amp = False      # CPU不支持AMP
    
    # 启动训练
    callbacks = Callbacks()
    train(hyp=opt.hyp, opt=opt, device=device, callbacks=callbacks)
    