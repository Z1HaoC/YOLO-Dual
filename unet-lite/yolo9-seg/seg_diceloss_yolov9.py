import argparse
import math
import os
import random
import sys
import time
import logging
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Callable
import json

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
import thop
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 初始化日志
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# 解决PIL版本兼容性问题
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

# 路径配置
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 项目根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 添加到环境变量
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 转换为相对路径

# 工具导入 - 确保这些模块存在或调整路径
try:
    import val_diceloss as validate  # 验证与评估（Dice损失版本）
    from utils.callbacks import Callbacks
    from utils.downloads import attempt_download
    from utils.general import (LOGGER as utils_LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, 
                           check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           increment_path, init_seeds, intersect_dicts, one_cycle, yaml_save, strip_optimizer)
    from utils.loggers import Loggers
    from utils.metrics import fitness
    from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, 
                                  smart_DDP, smart_optimizer, smart_resume, torch_distributed_zero_first)
    
    # 使用utils中的LOGGER
    LOGGER = utils_LOGGER
except ImportError as e:
    LOGGER.error(f"导入工具模块失败: {e}")
    sys.exit(1)

# 分布式训练配置
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

# CamVid数据集配置 - 12类，与ResNet50风格head匹配
CAMVID_COLORS = [
    [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
    [60, 40, 222], [128, 128, 0], [192, 128, 128], [64, 64, 128],
    [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 0]  # 最后一个是unlabelled
]

CLASS_NAMES = [
    "sky", "building", "pole", "road", "pavement",
    "tree", "signsymbol", "fence", "car", "pedestrian",
    "bicyclist", "unlabelled"
]
NUM_CLASSES = len(CLASS_NAMES)  # 保持12类，与新配置匹配


# -------------------------- 核心工具函数 --------------------------
def autopad(k: int, p: Optional[int] = None) -> int:
    """自动计算padding以保持特征图尺寸"""
    if p is None:
        p = k // 2 if k % 2 == 0 else (k - 1) // 2
    return p


# -------------------------- 数据增强模块 --------------------------
class DataAugmenter:
    """数据增强组合类"""
    def __init__(self, hyp: Dict):
        self.augmentations = [
            RandomHorizontalFlip(p=hyp.get('fliplr', 0.5)),
            RandomVerticalFlip(p=hyp.get('flipud', 0.2)),
            RandomRotation(degrees=hyp.get('degrees', 15), p=0.3),
            RandomBrightness(factor_range=(0.7, 1.3), p=0.3),
            RandomContrast(factor_range=(0.7, 1.3), p=0.3),
            RandomGaussianBlur(p=0.1),
            RandomCrop(scale_range=(0.7, 1.0), p=0.3)
        ]

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """应用随机数据增强"""
        for aug in self.augmentations:
            img, mask = aug(img, mask)
        return img, mask


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            return ImageOps.mirror(img), ImageOps.mirror(mask)
        return img, mask


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            return ImageOps.flip(img), ImageOps.flip(mask)
        return img, mask


class RandomRotation:
    def __init__(self, degrees: float = 15, p: float = 0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            img = img.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle, resample=Image.NEAREST)
        return img, mask


class RandomBrightness:
    def __init__(self, factor_range: Tuple[float, float] = (0.7, 1.3), p: float = 0.5):
        self.factor_range = factor_range
        self.p = p

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            factor = random.uniform(*self.factor_range)
            img = ImageEnhance.Brightness(img).enhance(factor)
        return img, mask


class RandomContrast:
    def __init__(self, factor_range: Tuple[float, float] = (0.7, 1.3), p: float = 0.5):
        self.factor_range = factor_range
        self.p = p

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            factor = random.uniform(*self.factor_range)
            img = ImageEnhance.Contrast(img).enhance(factor)
        return img, mask


class RandomGaussianBlur:
    def __init__(self, radius_range: Tuple[float, float] = (0.5, 2.0), p: float = 0.2):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img, mask


class RandomCrop:
    def __init__(self, scale_range: Tuple[float, float] = (0.7, 1.0), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            w, h = img.size
            scale = random.uniform(*self.scale_range)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            
            x1 = random.randint(0, w - new_w)
            y1 = random.randint(0, h - new_h)
            
            img = img.crop((x1, y1, x1 + new_w, y1 + new_h))
            mask = mask.crop((x1, y1, x1 + new_w, y1 + new_h))
            
            img = img.resize((w, h), Image.BILINEAR)
            mask = mask.resize((w, h), Image.NEAREST)
        return img, mask


# -------------------------- 数据集与数据加载 --------------------------
class JSONSegmentDataset(Dataset):
    """基于JSON掩码的语义分割数据集"""
    def __init__(self, img_dir: str, json_label_dir: str, img_size: int = 640, 
                 augment: bool = False, hyp: Optional[Dict] = None):
        self.img_dir = Path(img_dir)
        self.json_label_dir = Path(json_label_dir)
        self.img_size = img_size
        self.augment = augment
        self.augmenter = DataAugmenter(hyp) if (augment and hyp) else None
        
        # 获取图像文件列表
        self.img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.img_files = [f for f in self.img_dir.iterdir() if f.suffix.lower() in self.img_extensions]
        
        if not self.img_files:
            raise FileNotFoundError(f"在 {img_dir} 中未找到任何图像文件")
        
        # 验证并获取JSON掩码文件
        self.json_files = []
        for img_file in self.img_files:
            json_file = self.json_label_dir / f"{img_file.stem}.json"
            if not json_file.exists():
                raise FileNotFoundError(f"JSON掩码文件不存在: {json_file}")
            self.json_files.append(json_file)
        
        # 验证类别信息 - 强制使用12类以匹配新配置
        self.class_names = CLASS_NAMES
        self.num_classes = NUM_CLASSES
        LOGGER.info(f"加载数据集: {len(self.img_files)} 张图像, {self.num_classes} 个类别, 增强: {augment}")

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        # 加载图像和掩码
        img_path = self.img_files[index]
        json_path = self.json_files[index]
        
        try:
            img = Image.open(img_path).convert('RGB')
            with open(json_path, "r") as f:
                json_data = json.load(f)
            
            # 解析掩码数据
            mask_shape = json_data["shape"]
            mask_flat = np.array(json_data["mask_data"], dtype=np.int64)
            mask_np = mask_flat.reshape(mask_shape)
            mask_np = np.clip(mask_np, 0, self.num_classes - 1)  # 确保不超过12类
            mask = Image.fromarray(mask_np.astype(np.uint8))
        except Exception as e:
            LOGGER.error(f"加载数据失败 (索引 {index}): {e}")
            # 返回空白数据避免训练中断
            img = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
            mask = Image.new('L', (self.img_size, self.img_size), 0)
        
        # 应用数据增强
        if self.augment and self.augmenter:
            img, mask = self.augmenter(img, mask)
        
        # 调整尺寸并填充
        img, mask = self._resize_and_pad(img, mask)
        
        # 转换为张量
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).long()
        
        return img, mask, str(img_path)

    def _resize_and_pad(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """调整图像尺寸并填充至目标大小"""
        w, h = img.size
        scale = min(self.img_size / w, self.img_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 缩放
        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        
        # 计算填充
        pad_w = self.img_size - new_w
        pad_h = self.img_size - new_h
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        
        # 填充
        new_img = Image.new('RGB', (self.img_size, self.img_size), (128, 128, 128))
        new_mask = Image.new('L', (self.img_size, self.img_size), 0)
        
        new_img.paste(img, (pad_left, pad_top))
        new_mask.paste(mask, (pad_left, pad_top))
        
        return new_img, new_mask


def create_dataloader(img_dir: str, json_label_dir: str, img_size: int = 640, 
                     batch_size: int = 16, augment: bool = False, 
                     workers: int = 8, shuffle: bool = True, 
                     hyp: Optional[Dict] = None) -> Tuple[DataLoader, JSONSegmentDataset]:
    """创建数据加载器"""
    try:
        dataset = JSONSegmentDataset(
            img_dir=img_dir,
            json_label_dir=json_label_dir,
            img_size=img_size,
            augment=augment,
            hyp=hyp
        )
    except Exception as e:
        LOGGER.error(f"创建数据集失败: {e}")
        raise
    
    # 调整批次大小以适应数据集规模
    if batch_size > len(dataset):
        batch_size = len(dataset)
        LOGGER.warning(f"批次大小调整为 {batch_size}（数据集过小）")
    
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
            drop_last=True
        )
    except Exception as e:
        LOGGER.error(f"创建数据加载器失败: {e}")
        raise
    
    return dataloader, dataset


# 确保此函数在JSONSegmentDataset和create_dataloader之后定义
def create_json_segment_dataloader(img_dir: str, json_label_dir: str, img_size: int = 640, 
                                 batch_size: int = 16, augment: bool = False, 
                                 workers: int = 8, shuffle: bool = True, 
                                 hyp: Optional[Dict] = None) -> Tuple[DataLoader, JSONSegmentDataset]:
    """create_dataloader的别名，用于解决导入问题"""
    return create_dataloader(img_dir, json_label_dir, img_size, batch_size, 
                            augment, workers, shuffle, hyp)


# -------------------------- JSON掩码处理工具 --------------------------
def mask_to_json(mask_path: Path, json_save_path: Path) -> bool:
    """将掩码图像转换为JSON格式"""
    try:
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask, dtype=np.int64)
        mask_np = np.clip(mask_np, 0, NUM_CLASSES - 1)  # 强制限制为12类
        
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


def batch_convert_masks_to_json(mask_dir: str, json_save_dir: str) -> Tuple[int, int]:
    """批量转换掩码为JSON"""
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
    return success_count, len(mask_paths)


def verify_json_masks(img_dir: str, json_mask_dir: str) -> bool:
    """验证图像与JSON掩码的匹配性"""
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
        LOGGER.warning(f"发现 {len(missing)} 个图像缺少对应的JSON掩码")
        return False
    else:
        LOGGER.info(f"所有 {len(img_files)} 个图像都有对应的JSON掩码")
        return True


# -------------------------- YOLOv9核心网络模块 - 适配ResNet50风格 --------------------------
class Conv(nn.Module):
    """标准卷积模块: Conv2d -> BatchNorm2d -> SiLU"""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None, 
                 g: int = 1, act: bool = True):
        super().__init__()
        # 参数校验 - 特别注意通道数匹配ResNet50风格
        if not all(isinstance(x, int) for x in [c1, c2, k, s, g]):
            raise TypeError(f"Conv参数必须为整数, 实际: c1={type(c1)}, c2={type(c2)}")
        if g <= 0 or c1 % g != 0:
            raise ValueError(f"分组数g={g}必须为正整数且能整除c1={c1}")

        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入校验 - 确保通道数匹配
        if x is None:
            raise RuntimeError("Conv模块输入为None!")
        if not isinstance(x, torch.Tensor):
            raise RuntimeError(f"Conv输入必须是Tensor, 实际: {type(x)}")
        
        # 检查卷积层是否已正确初始化
        if self.conv is None:
            raise RuntimeError("Conv模块的conv层未初始化!")
            
        # 通道不匹配检查 - 对ResNet50风格尤为重要
        if x.shape[1] != self.conv.in_channels:
            raise RuntimeError(
                f"通道不匹配: 输入{x.shape[1]} vs 期望{self.conv.in_channels}"
            )
        
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class C3k2(nn.Module):
    """YOLOv9 C3k2模块, 带尺寸对齐保障 - 适配ResNet50深度"""
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Conv(c_, c_, 3, 1, g=g) for _ in range(n)))  # 3x3卷积保持尺寸
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)

        # 强制尺寸对齐 - 对ResNet50风格融合至关重要
        if y1.shape[2:] != y2.shape[2:]:
            target_h, target_w = min(y1.shape[2], y2.shape[2]), min(y1.shape[3], y2.shape[3])
            y1 = y1[:, :, :target_h, :target_w]
            y2 = y2[:, :, :target_h, :target_w]

        return self.cv3(torch.cat((y1, y2), 1)) + (x if self.add else 0)


class GAM(nn.Module):
    """全局聚合模块(Global Aggregation Module), 确保尺寸匹配 - 增强ResNet50特征提取"""
    def __init__(self, c: int, k: int = 1, s: int = 1, e: float = 0.25):
        super().__init__()
        c_ = int(c * e)
        
        # 1x1卷积降维 - 适配ResNet50通道
        self.conv1 = Conv(c, c_, k, s)
        
        # 全局池化 + 1x1卷积生成注意力权重
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = Conv(c_, c, k, s, act=False)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv3 = Conv(c_, c, k, s, act=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 记录输入尺寸 - 确保与ResNet50特征图匹配
        b, c, h, w = x.shape
        
        # 平均池化路径
        y1 = self.conv1(x)
        y1 = self.avg_pool(y1)
        y1 = self.conv2(y1)
        
        # 最大池化路径
        y2 = self.conv1(x)
        y2 = self.max_pool(y2)
        y2 = self.conv3(y2)
        
        # 融合并上采样至输入尺寸
        y = self.sigmoid(y1 + y2)
        y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
        
        return x * y


class SPPF(nn.Module):
    """空间金字塔池化-快速版 - 适配ResNet50高维特征"""
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Concat(nn.Module):
    """特征拼接模块, 自动对齐尺寸 - 对ResNet50风格解码器至关重要"""
    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        if len(x) <= 1:
            return x[0] if x else None
            
        # 以第一个张量为基准对齐尺寸 - 确保ResNet50特征图正确融合
        target_size = x[0].shape[2:]
        aligned = []
        for tensor in x:
            if tensor.shape[2:] != target_size:
                tensor = F.interpolate(
                    tensor, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            aligned.append(tensor)
            
        return torch.cat(aligned, self.d)


class C2f(nn.Module):
    """YOLOv8 C2f模块, 用于解码器 - 适配ResNet50风格"""
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Conv(self.c, self.c, 3, 1, g=g) for _ in range(n))
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1)) + (x if self.add else 0)


class C3(nn.Module):
    """C3模块, 用于解码器 - 与ResNet50风格head匹配"""
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Conv(c_, c_, 3, 1, g=g) for _ in range(n)))
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)) + (x if self.add else 0)


# -------------------------- YOLOv9语义分割模型 - 适配ResNet50风格head --------------------------
class YOLOv9Seg(nn.Module):
    """YOLOv9语义分割模型 - 适配ResNet50风格head结构"""
    def __init__(self, cfg: Dict, num_classes: Optional[int] = None):
        super().__init__()
        # 解析配置
        if isinstance(cfg, str):
            cfg = check_yaml(cfg)
            with open(cfg, 'r') as f:
                self.yaml = yaml.safe_load(f)
        else:
            self.yaml = cfg
        
        # 配置参数 - 确保与ResNet50风格配置匹配
        self.num_classes = self.yaml['nc'] if num_classes is None else num_classes
        self.yaml['nc'] = self.num_classes
        self.img_size = [640, 640]  # 与ResNet50常用输入尺寸匹配
        self.stride = torch.tensor([2, 4, 8, 16, 32])  # 下采样倍数，与ResNet50一致
        
        # 构建网络 - 使用新的backbone和head配置
        self.backbone, self.backbone_out_chs = self._build_backbone(self.yaml['backbone'])
        self.head, self.head_out_chs = self._build_head(self.yaml['head'], self.backbone_out_chs)
        
        # 初始化
        self._initialize_weights()
        self._log_model_info()

    def _build_backbone(self, cfg: List) -> Tuple[nn.ModuleList, List[int]]:
        """构建主干网络 - 适配ResNet50风格通道和深度"""
        backbone = nn.ModuleList()
        out_chs = []
        prev_out_ch = 3  # 输入通道(RGB)

        for layer_idx, layer in enumerate(cfg):
            from_, num_, module, args = layer
            # 计算输入通道 - 对ResNet50风格尤为重要
            c1 = prev_out_ch if from_ == -1 else out_chs[from_]

            # 构建模块 - 严格按照新配置解析
            if module == 'Conv':
                c2 = args[0]
                layer_module = Conv(c1, *args)
            elif module == 'C3k2':
                c2 = args[0]
                layer_module = C3k2(c1, *args)
            elif module == 'GAM':
                c2 = c1  # GAM不改变通道数
                layer_module = GAM(c1, *args)
            elif module == 'SPPF':
                c2 = args[0]
                layer_module = SPPF(c1, *args)
            elif module == 'Upsample':
                c2 = c1
                layer_module = self._create_upsample(*args)
            else:
                raise NotImplementedError(f"未知backbone模块: {module}")

            backbone.append(layer_module)
            out_chs.append(c2)
            prev_out_ch = c2
            LOGGER.debug(f"Backbone层 {layer_idx}: {module} ({c1}→{c2}) - 适配ResNet50风格")

        return backbone, out_chs

    def _build_head(self, cfg: List, backbone_out_chs: List[int]) -> Tuple[nn.ModuleList, List[int]]:
        """构建分割头 - 严格匹配ResNet50风格结构"""
        head = nn.ModuleList()
        all_out_chs = backbone_out_chs.copy()

        for i, layer in enumerate(cfg):
            from_, num_, module, args = layer
            # 计算输入通道 - 对ResNet50风格解码器至关重要
            if isinstance(from_, list):
                input_chs = [all_out_chs[f] for f in from_]
                c1 = sum(input_chs)
            else:
                c1 = all_out_chs[from_]

            # 构建模块 - 严格按照新配置解析
            if module == 'Conv':
                c2 = args[0]
                layer_module = Conv(c1, *args)
            elif module == 'Upsample':
                c2 = c1
                layer_module = self._create_upsample(*args)
            elif module == 'Concat':
                c2 = c1
                layer_module = Concat(*args)
            elif module == 'C2f':
                c2 = args[0]
                layer_module = C2f(c1, *args)
            elif module == 'C3':
                c2 = args[0]
                layer_module = C3(c1, *args)
            elif module == 'nn.Softmax':
                c2 = c1
                layer_module = nn.Softmax(*args)
            else:
                raise NotImplementedError(f"未知head模块: {module}")

            head.append(layer_module)
            all_out_chs.append(c2)
            LOGGER.debug(f"Head层 {i}: {module} ({c1}→{c2}) - 匹配ResNet50风格")

        return head, all_out_chs

    def _create_upsample(self, size_arg=None, scale_factor_arg=2.0, mode='nearest') -> nn.Upsample:
        """创建上采样层 - 确保与ResNet50特征图尺寸匹配"""
        size = None
        scale_factor = None
        
        # 解析尺寸参数
        if size_arg is not None:
            try:
                if isinstance(size_arg, (list, tuple)):
                    size = tuple(map(int, size_arg[-2:]))
                else:
                    size_val = int(size_arg)
                    size = (size_val, size_val)
            except:
                size = None
        
        # 解析缩放因子 - 对ResNet50风格的多尺度融合很重要
        if scale_factor is None and scale_factor_arg is not None:
            try:
                if isinstance(scale_factor_arg, (list, tuple)):
                    scale_factor = tuple(map(float, scale_factor_arg[-2:]))
                else:
                    scale_factor = float(scale_factor_arg)
            except:
                scale_factor = 2.0  # 默认值，与ResNet50上采样倍数一致

        # 创建上采样层
        upsample_kwargs = {
            'mode': mode,
            'align_corners': False if mode in ['bilinear', 'bicubic'] else None
        }
        if size is not None:
            upsample_kwargs['size'] = size
        else:
            upsample_kwargs['scale_factor'] = scale_factor

        return nn.Upsample(**upsample_kwargs)

    def forward(self, x: torch.Tensor, augment: bool = False, profile: bool = False) -> torch.Tensor:
        """前向传播"""
        return self._forward_once(x, profile)

    def _forward_once(self, x: torch.Tensor, profile: bool = False) -> torch.Tensor:
        """单次前向传播 - 确保ResNet50风格特征正确流动"""
        # 输入校验
        if x is None or not isinstance(x, torch.Tensor):
            raise RuntimeError(f"无效输入: {type(x)}")
        if x.dim() != 4:
            raise RuntimeError(f"输入必须是4维张量, 实际: {x.dim()}维")

        # Backbone传播 - 提取ResNet50风格特征
        x_outs = []
        for layer in self.backbone:
            x = layer(x)
            x_outs.append(x)
            if x is None:
                raise RuntimeError("Backbone输出为None")
        
        # Head传播 - 按照ResNet50风格解码器处理
        for layer, layer_cfg in zip(self.head, self.yaml['head']):
            from_, num_, module, args = layer_cfg
            
            # 获取输入 - 对多尺度融合至关重要
            if isinstance(from_, list):
                inputs = [x_outs[f] for f in from_]
            else:
                inputs = x_outs[from_]
            
            # 前向计算
            if isinstance(layer, Concat):
                x = layer(inputs)
            else:
                x = layer(inputs)
            
            x_outs.append(x)
            if x is None:
                raise RuntimeError(f"Head层 {module} 输出为None")
        
        # 确保输出尺寸与输入一致
        if x.shape[2:] != (self.img_size[0], self.img_size[1]):
            x = F.interpolate(
                x, 
                size=self.img_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        return x

    def _initialize_weights(self) -> None:
        """权重初始化 - 针对ResNet50风格进行优化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用与ResNet类似的初始化方法
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _log_model_info(self) -> None:
        """打印模型信息 - 强调ResNet50风格适配"""
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 计算FLOPs - 添加错误处理
        try:
            input_tensor = torch.randn(1, 3, *self.img_size)
            # 确保模型在正确设备上
            if next(self.parameters()).is_cuda:
                input_tensor = input_tensor.cuda()
            flops, _ = thop.profile(deepcopy(self), inputs=(input_tensor,), verbose=False)
            flops_g = flops / 1e9
        except Exception as e:
            flops_g = -1
            LOGGER.warning(f"计算FLOPs失败: {e}")
        
        LOGGER.info(f"ResNet50风格模型信息: {n_params:,} 总参数, {n_trainable:,} 可训练参数")
        LOGGER.info(f"计算量: {flops_g:.2f} GFLOPs (输入尺寸: {self.img_size})")
        LOGGER.info(f"类别数: {self.num_classes} (CamVid 12类)")


# -------------------------- 损失函数 --------------------------
class SegmentationLoss(nn.Module):
    """语义分割损失(交叉熵+Dice) - 优化适配12类"""
    def __init__(self, num_classes: int = 12, label_smoothing: float = 0.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
        # 类别权重
        if class_weights is not None:
            self.class_weights = class_weights if isinstance(class_weights, torch.Tensor) else torch.tensor(class_weights)
        else:
            self.class_weights = torch.ones(num_classes)
            
        # 交叉熵损失
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=label_smoothing
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        """计算损失"""
        # 确保权重设备匹配
        if self.class_weights.device != pred.device:
            self.class_weights = self.class_weights.to(pred.device)
            self.cross_entropy.weight = nn.Parameter(self.class_weights)
            
        # 批次大小校验
        if pred.size(0) != target.size(0):
            raise ValueError(f"批次不匹配: 输出{pred.size(0)} vs 目标{target.size(0)}")
        
        # 尺寸对齐 - 对ResNet50风格多尺度输出尤为重要
        if pred.shape[2:] != target.shape[1:]:
            target = F.interpolate(
                target.unsqueeze(1).float(),
                size=pred.shape[2:],
                mode='nearest'
            ).squeeze(1).long()
        
        # 计算损失
        ce_loss = self.cross_entropy(pred, target)
        dice_loss = self._dice_loss(pred.softmax(1), self._one_hot_encode(target))
        total_loss = ce_loss + 0.5 * dice_loss  # 平衡两种损失
        
        return total_loss, [total_loss.item(), ce_loss.item(), dice_loss.item()]

    def _one_hot_encode(self, target: torch.Tensor) -> torch.Tensor:
        """标签独热编码 - 12类"""
        b, h, w = target.shape
        one_hot = torch.zeros(b, self.num_classes, h, w, device=target.device)
        return one_hot.scatter_(1, target.unsqueeze(1), 1.0)

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """计算Dice损失 - 优化12类分割"""
        weighted_pred = pred * self.class_weights.view(1, -1, 1, 1).to(pred.device)
        intersection = (weighted_pred * target).sum(dim=(2, 3))
        pred_sum = weighted_pred.sum(dim=(2, 3))
        target_sum = target.sum(dim=(2, 3))
        dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
        return 1.0 - dice.mean()


# -------------------------- 辅助函数 --------------------------
def seg_labels_to_class_weights(json_files: List[str], num_classes: int) -> torch.Tensor:
    """从JSON掩码计算类别权重 - 适配12类"""
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
            LOGGER.warning(f"处理 {file} 失败: {e}")
            continue
    
    weights = total / (num_classes * (class_counts + 1e-8))
    return torch.from_numpy(weights).float()


def parse_class_weights(weight_str: str, num_classes: int) -> Optional[torch.Tensor]:
    """解析类别权重参数 - 适配12类"""
    if not weight_str:
        return None
        
    # 从文件加载
    if os.path.exists(weight_str):
        try:
            with open(weight_str, 'r') as f:
                weights = yaml.safe_load(f)
            if isinstance(weights, dict):
                weight_list = [weights[cls_name] for cls_name in CLASS_NAMES[:num_classes]]
                return torch.tensor(weight_list, dtype=torch.float32)
            elif isinstance(weights, list) and len(weights) == num_classes:
                return torch.tensor(weights, dtype=torch.float32)
            else:
                raise ValueError("权重文件格式错误")
        except Exception as e:
            LOGGER.error(f"加载权重文件失败: {e}")
            raise
    
    # 解析字符串
    try:
        weights = list(map(float, weight_str.split(',')))
        if len(weights) != num_classes:
            raise ValueError(f"权重数量不匹配: {len(weights)} vs {num_classes} (应为12)")
        return torch.tensor(weights, dtype=torch.float32)
    except Exception as e:
        LOGGER.error(f"解析权重失败: {e}")
        raise


def mask_to_rgb(mask: np.ndarray, color_map: List[List[int]]) -> np.ndarray:
    """掩码转RGB图像 - 12类配色"""
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(len(color_map)):
        rgb_mask[mask == cls] = color_map[cls]
    return rgb_mask


def visualize_prediction_difference(gt_mask: np.ndarray, pred_mask: np.ndarray, color_map: List[List[int]]) -> np.ndarray:
    """可视化预测差异 - 突出12类分割效果"""
    h, w = gt_mask.shape
    diff_mask = np.zeros((h, w, 3), dtype=np.uint8)
    diff_regions = gt_mask != pred_mask
    diff_mask[diff_regions] = [255, 0, 255]  # 洋红色标记差异
    
    for cls in range(len(color_map)):
        correct_regions = (gt_mask == cls) & (pred_mask == cls)
        diff_mask[correct_regions] = color_map[cls]
    
    return diff_mask


# -------------------------- 训练主函数 --------------------------
def train(hyp: Dict, opt: argparse.Namespace, device: torch.device, callbacks: Callbacks) -> Tuple[float, ...]:
    # 基础配置
    if device.type == 'cuda':
        torch.cuda.empty_cache()  # 清空缓存
        # 设置内存分配策略
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    save_dir = Path(opt.save_dir)
    epochs, batch_size, weights = opt.epochs, opt.batch_size, opt.weights
    resume, noval, nosave = opt.resume, opt.noval, opt.nosave
    workers, freeze = opt.workers, opt.freeze
    use_half = opt.half  # 半精度训练标志
    
    # 创建保存目录
    w = save_dir / 'weights'
    w.mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'

    # 处理超参数
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)
    
    hyp_defaults = {
        'lr0': 0.01, 'lrf': 0.2, 'momentum': 0.937, 'weight_decay': 0.0005,
        'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1,
        'label_smoothing': 0.0,
        'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
        'degrees': 10.0, 'translate': 0.1, 'scale': 0.5, 'shear': 2.0,
        'perspective': 0.001, 'flipud': 0.2, 'fliplr': 0.5
    }
    
    for k, v in hyp_defaults.items():
        if k not in hyp:
            hyp[k] = v
            LOGGER.info(f"补充超参数: {k} = {v}")

    LOGGER.info(colorstr('超参数: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
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
        
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
        if not isinstance(callbacks, Callbacks):
            callbacks = Callbacks()
        
        # 注册回调
        for k in dir(loggers):
            if k.startswith('on_') and callable(getattr(loggers, k, None)):
                try:
                    callbacks.register_action(k, getattr(loggers, k))
                except Exception as e:
                    LOGGER.warning(f"注册回调 {k} 失败: {e}")
        
        data_dict = loggers.remote_dataset

    # 数据集配置
    seed_value = (opt.seed + RANK) % (2**32)
    init_seeds(seed=seed_value, deterministic=False)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(opt.data)
    
    # 数据路径
    train_img_dir = data_dict.get('train_img', data_dict.get('train', ''))
    train_mask_dir = data_dict.get('train_label', os.path.join(data_dict.get('train', ''), 'masks'))
    train_json_dir = data_dict.get('train_json', os.path.join(train_img_dir, 'json'))
    val_img_dir = data_dict.get('val_img', data_dict.get('val', ''))
    val_mask_dir = data_dict.get('val_label', os.path.join(data_dict.get('val', ''), 'masks'))
    val_json_dir = data_dict.get('val_json', os.path.join(val_img_dir, 'json'))
    num_classes = 1 if opt.single_cls else int(data_dict['nc'])
    names = data_dict['names']

    # 验证并转换JSON掩码
    LOGGER.info("\n===== 验证JSON文件 =====")
    # 训练集处理
    if not verify_json_masks(train_img_dir, train_json_dir):
        LOGGER.info("训练集JSON不完整，开始转换...")
        train_success, train_total = batch_convert_masks_to_json(train_mask_dir, train_json_dir)
        if train_success == 0 and train_total > 0:
            LOGGER.error("训练集转换失败，终止")
            return (0.0,)
        if not verify_json_masks(train_img_dir, train_json_dir):
            LOGGER.error("训练集验证失败，终止")
            return (0.0,)
    
    # 验证集处理
    if not verify_json_masks(val_img_dir, val_json_dir):
        LOGGER.info("验证集JSON不完整，开始转换...")
        val_success, val_total = batch_convert_masks_to_json(val_mask_dir, val_json_dir)
        if val_success == 0 and val_total > 0:
            LOGGER.error("验证集转换失败，终止")
            return (0.0,)
        if not verify_json_masks(val_img_dir, val_json_dir):
            LOGGER.error("验证集验证失败，终止")
            return (0.0,)

    # 验证数据路径
    for path in [train_img_dir, train_json_dir, val_img_dir, val_json_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据路径不存在: {path}")
        LOGGER.info(f"找到数据路径: {path}")

    # 模型初始化 - 使用ResNet50风格配置
    check_suffix(weights, '.pt')
    pretrained = weights.endswith('.pt')
    try:
        model = YOLOv9Seg(opt.cfg, num_classes=num_classes).to(device)
        
        # 根据参数设置模型精度
        if use_half:
            model = model.half()
            LOGGER.info("启用半精度模型")
    except Exception as e:
        LOGGER.error(f"模型初始化失败: {e}")
        raise
    
    # 加载预训练权重
    if pretrained:
        weights = attempt_download(weights)
        try:
            ckpt = torch.load(weights, map_location='cpu')
            exclude = []
            csd = ckpt['model'].float().state_dict() if 'model' in ckpt else ckpt.float().state_dict()
            csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
            model.load_state_dict(csd, strict=False)
            LOGGER.info(f'加载权重: {len(csd)}/{len(model.state_dict())} 项匹配')
        except Exception as e:
            LOGGER.error(f"加载权重失败: {e}")
            raise

    # 冻结层 - 针对ResNet50风格backbone调整
    freeze = [f'backbone.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = not any(x in k for x in freeze)
        if not v.requires_grad:
            LOGGER.info(f'冻结层: {k}')

    # 图像尺寸调整（32的倍数）
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)
    if imgsz % 32 != 0:
        imgsz = ((imgsz // 32) + 1) * 32
        LOGGER.warning(f"输入尺寸调整为32的倍数: {imgsz}")

    # 自动批量大小
    if RANK == -1 and batch_size == -1:
        batch_size = 16

    # 优化器与学习率 - 可针对ResNet50风格调整
    nbs = 64
    accumulate = 1
    hyp['weight_decay'] *= batch_size * accumulate / nbs
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # 学习率调度
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

    # 数据加载器
    try:
        train_loader, dataset = create_dataloader(
            img_dir=train_img_dir,
            json_label_dir=train_json_dir,
            img_size=imgsz,
            batch_size=batch_size // WORLD_SIZE,
            augment=True,
            workers=workers,
            hyp=hyp
        )
    except Exception as e:
        LOGGER.error(f"创建训练数据加载器失败: {e}")
        raise
    
    val_loader = None
    if RANK in {-1, 0}:
        try:
            val_loader = create_dataloader(
                img_dir=val_img_dir,
                json_label_dir=val_json_dir,
                img_size=imgsz,
                batch_size=batch_size // WORLD_SIZE * 2,
                augment=False,
                workers=workers*2,
                shuffle=False,
                hyp=hyp
            )[0]
        except Exception as e:
            LOGGER.error(f"创建验证数据加载器失败: {e}")
            raise

    # 模型属性
    model.num_classes = num_classes
    model.hyp = hyp
    model.names = names

    # 类别权重 - 12类
    if opt.class_weights:
        class_weights = parse_class_weights(opt.class_weights, num_classes)
        LOGGER.info(f"使用自定义类别权重: {class_weights.tolist()}")
    else:
        class_weights = seg_labels_to_class_weights(dataset.json_files, num_classes)
        LOGGER.info(f"使用数据驱动权重: {class_weights.tolist()}")
    
    # 调整类别权重精度以匹配模型
    class_weights = class_weights.to(device)
    if use_half:
        class_weights = class_weights.half()
    model.class_weights = class_weights

    # 损失函数
    criterion = SegmentationLoss(
        num_classes=num_classes, 
        label_smoothing=hyp['label_smoothing'],
        class_weights=class_weights
    )
    
    # AMP配置
    amp_enabled = opt.amp and cuda
    scaler = None
    if amp_enabled:
        try:
            from torch.cuda import amp
            scaler = amp.GradScaler(enabled=amp_enabled)
            LOGGER.info("启用AMP混合精度训练")
        except ImportError:
            LOGGER.warning("PyTorch版本不支持AMP，禁用")
            amp_enabled = False
    else:
        LOGGER.info("禁用AMP")

    # 半精度与AMP的兼容性检查
    if use_half and amp_enabled:
        LOGGER.warning("半精度训练与AMP同时启用，可能导致冲突，建议只使用其中一种")

    stopper = EarlyStopping(patience=opt.patience)

    # 训练循环
    LOGGER.info(colorstr('开始训练（YOLOv9语义分割 - ResNet50风格head）!'))
    t0 = time.time()
    global_step = 0
    
    for epoch in range(start_epoch, epochs):
        model.train()
        mloss = torch.zeros(3, device=device)  # 总损失, CE损失, Dice损失
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 6) % ('Epoch', 'gpu_mem', '总损失', 'CE损失', 'Dice损失', 'img_size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=len(train_loader), bar_format=TQDM_BAR_FORMAT)
        
        for i, (imgs, targets, paths) in pbar:
            # 根据半精度参数调整输入数据类型
            imgs = imgs.to(device, non_blocking=True).float()
            if use_half:
                imgs = imgs.half()
            
            targets = targets.to(device, non_blocking=True).long()

            # 前向传播
            try:
                if amp_enabled:
                    with amp.autocast():
                        pred = model(imgs)
                        loss, loss_items = criterion(pred, targets)
                        loss_items = torch.tensor(loss_items, device=device)
                else:
                    pred = model(imgs)
                    loss, loss_items = criterion(pred, targets)
                    loss_items = torch.tensor(loss_items, device=device)
            except Exception as e:
                LOGGER.error(f"前向传播失败 (批次 {i}): {e}")
                continue

            # 反向传播
            try:
                if amp_enabled:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            except Exception as e:
                LOGGER.error(f"反向传播失败 (批次 {i}): {e}")
                continue

            # 梯度更新
            if (i + 1) % accumulate == 0 or i == len(train_loader) - 1:
                try:
                    if amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)
                except Exception as e:
                    LOGGER.error(f"梯度更新失败 (批次 {i}): {e}")
                    continue

            # 日志
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = f'{torch.cuda.memory_reserved() / 1E9:.3g}G' if cuda else 'N/A'
                pbar.set_description(('%10s' * 2 + '%10.4g' * 4) %
                                    (f'{epoch}/{epochs - 1}', mem, *mloss, imgs.shape[-1]))

                # TensorBoard
                if global_step % 10 == 0:
                    tb_writer.add_scalar('Train/Total_Loss', loss_items[0], global_step)
                    tb_writer.add_scalar('Train/CE_Loss', loss_items[1], global_step)
                    tb_writer.add_scalar('Train/Dice_Loss', loss_items[2], global_step)
                    tb_writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)

                # 可视化
                if global_step % 100 == 0 and global_step != 0:
                    num_samples = min(3, imgs.size(0))
                    for s in range(num_samples):
                        img_np = imgs[s].permute(1, 2, 0).cpu().detach().numpy()
                        target_np = targets[s].cpu().detach().numpy()
                        pred_np = torch.argmax(pred[s], dim=0).cpu().detach().numpy()

                        target_rgb = mask_to_rgb(target_np, CAMVID_COLORS)
                        pred_rgb = mask_to_rgb(pred_np, CAMVID_COLORS)
                        diff_rgb = visualize_prediction_difference(target_np, pred_np, CAMVID_COLORS)

                        tb_writer.add_image(f'Train/Sample_{s}/Input', img_np, global_step, dataformats='HWC')
                        tb_writer.add_image(f'Train/Sample_{s}/GroundTruth', target_rgb, global_step, dataformats='HWC')
                        tb_writer.add_image(f'Train/Sample_{s}/Prediction', pred_rgb, global_step, dataformats='HWC')
                        tb_writer.add_image(f'Train/Sample_{s}/Difference', diff_rgb, global_step, dataformats='HWC')

                global_step += 1

        # 学习率更新
        scheduler.step()

        # 验证与保存
        if RANK in {-1, 0}:
            ema.update_attr(model, include=['yaml', 'num_classes', 'hyp', 'names', 'stride'])
            final_epoch = (epoch == epochs - 1)
            if not noval or final_epoch:
                try:
                    results, maps, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=ema.ema,
                        single_cls=opt.single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        verbose=True,
                        half=use_half  # 传递半精度参数到验证函数
                    )
                except Exception as e:
                    LOGGER.error(f"验证失败: {e}")
                    results = None

                # 验证日志
                if results is not None:
                    tb_writer.add_scalar('Validation/mIoU', results[0], epoch)
                    for cls_idx, cls_name in enumerate(names):
                        if cls_idx < len(maps):
                            tb_writer.add_scalar(f'Validation/IoU_{cls_name}', maps[cls_idx], epoch)

                    # 验证集可视化
                    model.eval()
                    with torch.no_grad():
                        try:
                            val_batch = next(iter(val_loader))
                            val_imgs, val_targets, _ = val_batch
                            val_imgs = val_imgs.to(device)
                            if use_half:
                                val_imgs = val_imgs.half()
                            val_preds = model(val_imgs)
                            val_preds = torch.argmax(val_preds, dim=1)
                            
                            num_val_samples = min(3, val_imgs.size(0))
                            for s in range(num_val_samples):
                                img_np = val_imgs[s].permute(1, 2, 0).cpu().detach().numpy()
                                target_np = val_targets[s].cpu().detach().numpy()
                                pred_np = val_preds[s].cpu().detach().numpy()
                                
                                target_rgb = mask_to_rgb(target_np, CAMVID_COLORS)
                                pred_rgb = mask_to_rgb(pred_np, CAMVID_COLORS)
                                diff_rgb = visualize_prediction_difference(target_np, pred_np, CAMVID_COLORS)
                                
                                tb_writer.add_image(f'Validation/Sample_{s}/Input', img_np, epoch, dataformats='HWC')
                                tb_writer.add_image(f'Validation/Sample_{s}/GroundTruth', target_rgb, epoch, dataformats='HWC')
                                tb_writer.add_image(f'Validation/Sample_{s}/Prediction', pred_rgb, epoch, dataformats='HWC')
                                tb_writer.add_image(f'Validation/Sample_{s}/Difference', diff_rgb, epoch, dataformats='HWC')
                        except Exception as e:
                            LOGGER.error(f"验证可视化失败: {e}")

            # 保存模型
            fi = fitness(np.array(results).reshape(1, -1)) if results else 0.0
            if fi > best_fitness:
                best_fitness = fi
            save = (not nosave) or (final_epoch and not opt.evolve)
            if save:
                try:
                    torch.save({
                        'model': ema.ema, 
                        'optimizer': optimizer.state_dict(), 
                        'epoch': epoch, 
                        'best_fitness': best_fitness
                    }, last)
                    if fi == best_fitness:
                        torch.save({'model': ema.ema}, best)
                        LOGGER.info(f"更新最佳模型 (mIoU: {results[0]:.4f})")
                except Exception as e:
                    LOGGER.error(f"保存模型失败: {e}")

            # 早停
            if stopper(epoch=epoch, fitness=fi):
                break

    # 训练结束
    if RANK in {-1, 0}:
        if tb_writer:
            tb_writer.close()
            
        LOGGER.info(f'\n训练完成 ({(time.time() - t0) / 3600:.2f} 小时)')
        LOGGER.info(f"最佳模型: {best}")
        strip_optimizer(best)

    return (best_fitness,)


# -------------------------- 命令行入口 --------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='初始权重路径')
    parser.add_argument('--cfg', type=str, default='yolov9_seg.yaml', help='模型配置文件 - 使用新的ResNet50风格配置')
    parser.add_argument('--data', type=str, default='/root/BestYOLO/CamVid/data.yaml', help='数据集配置')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=1, help='批量大小')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='输入图像尺寸 - 与ResNet50匹配')
    parser.add_argument('--device', default='0', help='设备, 如 0 或 cpu')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='冻结层索引')
    parser.add_argument('--patience', type=int, default=150, help='早停patience')
    parser.add_argument('--single-cls', action='store_true', help='单类别训练')
    parser.add_argument('--sync-bn', action='store_true', help='同步BN')
    parser.add_argument('--cos-lr', action='store_true', help='余弦学习率')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='恢复训练')
    parser.add_argument('--save-dir', type=str, default='runs/train-yolov9-seg-resnet50', help='保存目录')
    parser.add_argument('--optimizer', type=str, default='SGD', help='优化器')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='标签平滑')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--noval', action='store_true', help='禁用验证')
    parser.add_argument('--nosave', action='store_true', help='不保存模型')
    parser.add_argument('--hyp', type=str, default='/root/BestYOLO/data/hyps/hyp.scratch-seg.yaml', help='超参数文件')
    parser.add_argument('--exist-ok', action='store_true', help='允许覆盖实验目录')
    parser.add_argument('--amp', action='store_true', default=False, help='启用AMP')
    parser.add_argument('--class-weights', type=str, default='weight.yaml', help='类别权重, 如 "1.0,2.0" 或文件路径 (12类)')
    parser.add_argument('--noplots', action='store_true', help='禁用绘图功能')
    parser.add_argument('--evolve', action='store_true', help='启用超参数进化搜索')
    parser.add_argument('--half', action='store_true', help='使用半精度训练')
    opt = parser.parse_args()
    check_requirements(exclude=('tensorboard', 'thop'))
    
    # 设备选择
    device = select_device(opt.device, batch_size=opt.batch_size)
    if device.type == 'cpu':
        opt.sync_bn = False
        opt.amp = False
        if opt.half:
            LOGGER.warning("CPU不支持半精度训练，自动禁用")
            opt.half = False
    
    # 启动训练 - 使用ResNet50风格配置
    callbacks = Callbacks()
    try:
        train(hyp=opt.hyp, opt=opt, device=device, callbacks=callbacks)
    except Exception as e:
        LOGGER.error(f"训练过程失败: {e}")
        sys.exit(1)
