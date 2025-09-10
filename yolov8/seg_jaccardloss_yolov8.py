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
import val_jaccardloss as validate  # 验证与评估
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


# -------------------------- 语义分割数据集类 --------------------------
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


# -------------------------- YOLOv8 Backbone 核心模块 --------------------------
def autopad(k: int, p: Optional[int] = None) -> int:
    """自动计算卷积填充"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """YOLOv8标准卷积模块：Conv2d -> BatchNorm2d -> SiLU"""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None, 
                 g: int = 1, act: bool = True):
        super().__init__()
        if not all(isinstance(x, int) for x in [c1, c2, k, s, g]):
            raise TypeError(f"Conv模块参数必须为整数，当前参数: c1={c1}({type(c1)}), c2={c2}({type(c2)}), "
                           f"k={k}({type(k)}), s={s}({type(s)}), g={g}({type(g)})")
        if g <= 0 or c1 % g != 0:
            raise ValueError(f"分组数g={g}必须为正整数且能整除输入通道数c1={c1}")

        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()  # YOLOv8默认SiLU激活

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.conv.weight.shape[1]:
            raise RuntimeError(
                f"Conv层输入通道不匹配: 实际输入{x.shape[1]}，权重期望{self.conv.weight.shape[1]}\n"
                f"卷积层配置: in_channels={self.conv.weight.shape[1]}, out_channels={self.conv.weight.shape[0]}, "
                f"kernel_size={self.conv.kernel_size}"
            )
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class C2f(nn.Module):
    """YOLOv8的C2f模块，替代YOLOv5的C3模块"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 合并后的卷积
        self.m = nn.ModuleList(Conv(self.c, self.c, 3, 1, g=g) for _ in range(n))
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1)) + (x if self.add else 0)


class C2f_DCN(nn.Module):
    """带可变形卷积的C2f模块"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 合并后的卷积
        
        # 构建DCN子模块
        dcn_blocks = []
        try:
            from torchvision.ops import DeformConv2d
        except ImportError:
            raise ImportError("使用C2f-DCN需安装torchvision>=0.11.0，执行：pip install torchvision>=0.11.0")
        
        for _ in range(n):
            # 1. 普通卷积预处理
            conv = Conv(self.c, self.c, 3, 1, g=g, act=False)
            # 2. 偏移量生成层
            offset_conv = Conv(self.c, 2*3*3, 3, 1, g=g, act=True)
            # 3. 可变形卷积层
            dcn_conv = DeformConv2d(self.c, self.c, kernel_size=3, padding=1, groups=g, bias=False)
            # 4. BN+SiLU激活
            bn_silu = nn.Sequential(nn.BatchNorm2d(self.c), nn.SiLU(inplace=True))
            
            dcn_blocks.append(nn.Sequential(conv, offset_conv, dcn_conv, bn_silu))
        
        self.m = nn.ModuleList(dcn_blocks)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            # DCN块前向传播
            x1 = m[0](y[-1])          # 普通卷积预处理
            offset = m[1](x1)         # 生成偏移量
            x1 = m[2](x1, offset)     # 可变形卷积
            x1 = m[3](x1)             # BN+SiLU
            y.append(x1)
            
        return self.cv2(torch.cat(y, 1)) + (x if self.add else 0)


class SPPF(nn.Module):
    """YOLOv8空间金字塔池化-快速版"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Concat(nn.Module):
    """特征拼接模块"""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        if len(x) <= 1:
            return x[0] if x else None
            
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


# -------------------------- YOLOv8Seg 语义分割模型 --------------------------
class YOLOv8Seg(nn.Module):
    """完整的YOLOv8语义分割模型（编码器：YOLOv8 backbone，解码器：统一分割头）"""
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
        self.stride = torch.tensor([2, 4, 8, 16, 32])  # YOLOv8各阶段下采样倍数
        
        # 构建主干网络和分割头
        self.backbone, self.backbone_out_chs = self._build_backbone(self.yaml['backbone'])
        self.head, self.head_out_chs = self._build_head(self.yaml['head'], self.backbone_out_chs)
        
        # 初始化权重与日志
        self._initialize_weights()
        self._log_model_info()

    def _build_backbone(self, cfg):
        """构建YOLOv8 backbone（使用C2f模块）"""
        backbone = nn.ModuleList()
        out_chs = []
        prev_out_ch = 3  # 初始输入通道（RGB）

        for layer_idx, layer in enumerate(cfg):
            from_, num_, module, args = layer
            # 计算输入通道
            if from_ == -1:
                c1 = prev_out_ch
            else:
                c1 = out_chs[from_]

            # 构建模块
            if module == 'Conv':
                c2 = args[0]
                layer_module = Conv(c1, *args)
            elif module == 'C2f':  # 使用C2f替代C3
                c2 = args[0]
                layer_module = C2f(c1, *args)
            elif module == 'C2f_DCN':  # 带DCN的C2f模块
                c2 = args[0]
                layer_module = C2f_DCN(c1, *args)
            elif module == 'SPPF':
                c2 = args[0]
                layer_module = SPPF(c1, *args)
            else:
                raise NotImplementedError(f"未知模块: {module}")

            backbone.append(layer_module)
            out_chs.append(c2)
            prev_out_ch = c2
            LOGGER.debug(f"Backbone层 {layer_idx}: {module} → {c1}→{c2}通道")

        return backbone, out_chs

    def _build_head(self, cfg, backbone_out_chs):
            """构建统一分割头"""
            head = nn.ModuleList()
            all_out_chs = backbone_out_chs.copy()

            for i, layer in enumerate(cfg):
                from_, num_, module, args = layer
                # 1. 计算输入通道
                if isinstance(from_, list):
                    input_chs = [all_out_chs[f] for f in from_]
                    c1 = sum(input_chs)
                else:
                    c1 = all_out_chs[from_]

                # 2. 构建模块并计算输出通道
                if module == 'Conv':
                    c2 = args[0]
                    layer_module = Conv(c1, *args)
                elif module == 'nn.Upsample' or module == 'Upsample':
                    c2 = c1  # 上采样不改变通道数
                    # 解析参数
                    if len(args) >= 3:
                        size_arg, scale_factor_arg, mode = args[0], args[1], args[2]
                    else:
                        size_arg = None
                        scale_factor_arg = 2.0  # 确保默认是浮点数
                        mode = 'nearest'

                    # 处理size（必须是整数元组或None）
                    size = None
                    if size_arg is not None:
                        if isinstance(size_arg, (list, tuple)):
                            # 截取最后两个维度并转换为整数
                            size_list = [int(x) for x in size_arg[-2:]]
                            size = tuple(size_list) if len(size_list) >= 2 else (size_list[0], size_list[0])
                        elif isinstance(size_arg, (int, float)):
                            size = (int(size_arg), int(size_arg))
                        else:
                            try:
                                size_val = int(size_arg)
                                size = (size_val, size_val)
                            except:
                                size = (256, 256)  # 最终兜底值

                        # 确保size是2元素整数元组
                        if len(size) != 2:
                            size = (size[0], size[0])

                    # 处理scale_factor（必须是浮点数、浮点数元组或None）
                    scale_factor = None
                    if scale_factor_arg is not None:
                        if isinstance(scale_factor_arg, (list, tuple)):
                            # 截取最后两个维度并转换为浮点数
                            scale_list = [float(x) for x in scale_factor_arg[-2:]]
                            scale_factor = tuple(scale_list) if len(scale_list) >= 2 else (scale_list[0], scale_list[0])
                        elif isinstance(scale_factor_arg, (int, float)):
                            scale_factor = float(scale_factor_arg)  # 确保是浮点数
                        else:
                            try:
                                scale_factor = float(scale_factor_arg)
                            except:
                                scale_factor = 2.0  # 转换失败时使用默认浮点数

                    # 强制参数组合有效性
                    if size is not None:
                        # 有size时，必须清空scale_factor
                        scale_factor = None
                    else:
                        # 无size时，必须确保scale_factor有效
                        if scale_factor is None:
                            scale_factor = 2.0  # 默认缩放因子
                        # 确保scale_factor类型正确
                        if isinstance(scale_factor, tuple):
                            if any(not isinstance(x, float) for x in scale_factor):
                                scale_factor = (2.0, 2.0)
                        elif not isinstance(scale_factor, float):
                            scale_factor = 2.0

                    # 创建Upsample层
                    if mode == 'nearest':
                        # 对nearest模式进行额外校验
                        if size is not None:
                            layer_module = nn.Upsample(size=size, mode=mode)
                        else:
                            layer_module = nn.Upsample(scale_factor=scale_factor, mode=mode)
                    elif mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
                        layer_module = nn.Upsample(
                            size=size,
                            scale_factor=scale_factor,
                            mode=mode,
                            align_corners=False
                        )
                    else:
                        layer_module = nn.Upsample(size=size, scale_factor=scale_factor, mode=mode)

                elif module == 'Concat':
                    c2 = c1
                    layer_module = Concat(*args)
                elif module == 'C2f':  # 使用C2f替代C3
                    c2 = args[0]
                    layer_module = C2f(c1, *args)
                elif module == 'nn.Softmax':
                    c2 = c1
                    layer_module = nn.Softmax(*args)
                else:
                    raise NotImplementedError(f"YOLOv8分割头未知模块: {module}")

                head.append(layer_module)
                all_out_chs.append(c2)
                LOGGER.debug(f"YOLOv8 Head层 {i}: {module} → 输入{c1}→输出{c2}通道")

            return head, all_out_chs



    def forward(self, x: torch.Tensor, augment: bool = False, profile: bool = False) -> torch.Tensor:
        """前向传播"""
        return self._forward_once(x, profile)

    def _forward_once(self, x: torch.Tensor, profile: bool = False) -> torch.Tensor:
        """单次前向传播"""
        # 1. Backbone前向传播，保存所有中间特征
        x_outs = []
        for layer in self.backbone:
            x = layer(x)
            x_outs.append(x)
        
        # 2. Head前向传播，按from参数获取输入特征
        for layer, layer_cfg in zip(self.head, self.yaml['head']):
            from_, num_, module, args = layer_cfg
            
            # 获取当前层输入
            if isinstance(from_, list):
                inputs = [x_outs[f] for f in from_]
            else:
                inputs = x_outs[from_]  # 单输入从指定层获取
            
            # 执行层计算
            if isinstance(layer, Concat):
                x = layer(inputs)
            else:
                x = layer(inputs)
            
            # 保存当前层输出到特征列表
            x_outs.append(x)
        
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
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
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
        flops, _ = thop.profile(deepcopy(self), inputs=(input_tensor,), verbose=False)
        flops_g = flops / 1e9
        
        LOGGER.info(f"模型信息: {n_params:,} 总参数, {n_trainable:,} 可训练参数")
        LOGGER.info(f"计算量: {flops_g:.2f} GFLOPs (输入尺寸: {self.img_size})")
        LOGGER.info(f"类别数: {self.num_classes}")


# -------------------------- 辅助函数 --------------------------
def scale_img(img: torch.Tensor, ratio: float = 1.0, gs: int = 32) -> torch.Tensor:
    h, w = img.shape[2:]
    new_h = math.ceil(h * ratio / gs) * gs
    new_w = math.ceil(w * ratio / gs) * gs
    return F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)


class SegmentationLoss(nn.Module):
    """语义分割损失（交叉熵+Jaccard）"""
    def __init__(self, num_classes: int = 12, label_smoothing: float = 0.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
        # 设置类别权重，如果未提供则使用均匀权重
        if class_weights is not None:
            self.class_weights = class_weights
        else:
            self.class_weights = torch.ones(num_classes)
            
        # 使用带权重的交叉熵损失
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=label_smoothing
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        if pred.size(0) != target.size(0):
            raise ValueError(
                f"批次大小不匹配: 模型输出批次大小为{pred.size(0)}, "
                f"目标标签批次大小为{target.size(0)}"
            )
        
        if pred.shape[2:] != target.shape[1:]:
            target = F.interpolate(
                target.unsqueeze(1).float(),
                size=pred.shape[2:],
                mode='nearest'
            ).squeeze(1).long()
        
        ce_loss = self.cross_entropy(pred, target)
        jaccard_loss = self._jaccard_loss(pred.softmax(1), self._one_hot_encode(target))
        total_loss = ce_loss + 0.5 * jaccard_loss
        
        return total_loss, [total_loss.item(), ce_loss.item(), jaccard_loss.item()]

    def _one_hot_encode(self, target: torch.Tensor) -> torch.Tensor:
        b, h, w = target.shape
        one_hot = torch.zeros(b, self.num_classes, h, w, device=target.device)
        return one_hot.scatter_(1, target.unsqueeze(1), 1.0)

    def _jaccard_loss(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        计算Jaccard损失（IoU损失）
        Jaccard损失 = 1 - IoU
        IoU = 交集 / 并集 = (预测 ∩ 目标) / (预测 ∪ 目标)
        """
        # 对Jaccard损失应用类别权重
        weighted_pred = pred * self.class_weights.view(1, -1, 1, 1).to(pred.device)
        
        # 计算交集
        intersection = (weighted_pred * target).sum(dim=(2, 3))
        # 计算并集 = 预测 + 目标 - 交集
        union = weighted_pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        # 计算IoU
        jaccard = (intersection + eps) / (union + eps)
        # Jaccard损失 = 1 - 平均IoU
        return 1.0 - jaccard.mean()


def seg_labels_to_class_weights(json_files, num_classes):
    """计算类别权重（基于数据分布的自动权重）"""
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
    
    weights = total / (num_classes * (class_counts + 1e-8))
    return torch.from_numpy(weights).float()


def parse_class_weights(weight_str, num_classes):
    """解析命令行输入的类别权重字符串"""
    if not weight_str:
        return None
        
    # 尝试从文件加载权重
    if os.path.exists(weight_str):
        try:
            with open(weight_str, 'r') as f:
                weights = yaml.safe_load(f)
            # 支持字典格式 {class_name: weight}
            if isinstance(weights, dict):
                # 确保所有类别都有对应的权重
                weight_list = []
                for cls_name in CLASS_NAMES[:num_classes]:
                    if cls_name in weights:
                        weight_list.append(weights[cls_name])
                    else:
                        raise ValueError(f"类别权重文件中缺少类别 {cls_name} 的权重")
                return torch.tensor(weight_list, dtype=torch.float32)
            # 支持列表格式
            elif isinstance(weights, list):
                if len(weights) != num_classes:
                    raise ValueError(f"类别权重数量与类别数不匹配: {len(weights)} vs {num_classes}")
                return torch.tensor(weights, dtype=torch.float32)
            else:
                raise ValueError("类别权重文件格式不正确，应为字典或列表")
        except Exception as e:
            LOGGER.error(f"加载类别权重文件失败: {e}")
            raise
    
    # 解析逗号分隔的权重字符串
    try:
        weights = list(map(float, weight_str.split(',')))
        if len(weights) != num_classes:
            raise ValueError(f"类别权重数量与类别数不匹配: {len(weights)} vs {num_classes}")
        return torch.tensor(weights, dtype=torch.float32)
    except Exception as e:
        LOGGER.error(f"解析类别权重失败: {e}")
        raise


def mask_to_rgb(mask, color_map):
    """掩码转RGB"""
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(len(color_map)):
        rgb_mask[mask == cls] = color_map[cls]
    return rgb_mask


def visualize_prediction_difference(gt_mask, pred_mask, color_map):
    """可视化预测差异"""
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
            LOGGER.info(f"补充缺失的超参数: {k} = {v}")

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
            if k.startswith('on_') and not k.startswith('__'):
                callback = getattr(loggers, k, None)
                if callback is not None and callable(callback):
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
    
    train_img_dir = data_dict.get('train_img', data_dict.get('train', ''))
    train_mask_dir = data_dict.get('train_label', os.path.join(data_dict.get('train', ''), 'masks'))
    train_json_dir = data_dict.get('train_json', os.path.join(train_img_dir, 'json'))
    val_img_dir = data_dict.get('val_img', data_dict.get('val', ''))
    val_mask_dir = data_dict.get('val_label', os.path.join(data_dict.get('val', ''), 'masks'))
    val_json_dir = data_dict.get('val_json', os.path.join(val_img_dir, 'json'))
    num_classes = 1 if opt.single_cls else int(data_dict['nc'])
    names = data_dict['names']

    # 验证并转换JSON掩码
    LOGGER.info("\n===== 开始验证JSON文件 =====")
    train_json_valid = verify_json_masks(train_img_dir, train_json_dir)
    if not train_json_valid:
        LOGGER.info("训练集JSON文件不完整，开始转换掩码为JSON...")
        train_success, train_total = batch_convert_masks_to_json(train_mask_dir, train_json_dir)
        if train_success == 0 and train_total > 0:
            LOGGER.error("训练集掩码转换失败，无法继续训练")
            return (0.0,)
        train_json_valid = verify_json_masks(train_img_dir, train_json_dir)
        if not train_json_valid:
            LOGGER.error("训练集JSON文件验证仍失败，无法继续训练")
            return (0.0,)
    
    val_json_valid = verify_json_masks(val_img_dir, val_json_dir)
    if not val_json_valid:
        LOGGER.info("验证集JSON文件不完整，开始转换掩码为JSON...")
        val_success, val_total = batch_convert_masks_to_json(val_mask_dir, val_json_dir)
        if val_success == 0 and val_total > 0:
            LOGGER.error("验证集掩码转换失败，无法继续训练")
            return (0.0,)
        val_json_valid = verify_json_masks(val_img_dir, val_json_dir)
        if not val_json_valid:
            LOGGER.error("验证集JSON文件验证仍失败，无法继续训练")
            return (0.0,)

    # 验证数据路径
    for path in [train_img_dir, train_json_dir, val_img_dir, val_json_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据路径不存在: {path}")
        else:
            LOGGER.info(f"找到数据路径: {path}")

    # 模型初始化（使用YOLOv8Seg）
    check_suffix(weights, '.pt')
    pretrained = weights.endswith('.pt')
    model = YOLOv8Seg(opt.cfg, num_classes=num_classes).to(device)  # 使用YOLOv8Seg模型
    
    # 加载预训练权重
    if pretrained:
        weights = attempt_download(weights)
        ckpt = torch.load(weights, map_location='cpu')
        exclude = []
        csd = ckpt['model'].float().state_dict() if 'model' in ckpt else ckpt.float().state_dict()
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
        batch_size = 16

    # 优化器与学习率调度
    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / nbs
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

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
    train_loader, dataset = create_json_segment_dataloader(
        img_dir=train_img_dir,
        json_label_dir=train_json_dir,
        img_size=imgsz,
        batch_size=batch_size // WORLD_SIZE,
        augment=True,
        workers=workers,
        hyp=hyp
    )
    
    val_loader = None
    if RANK in {-1, 0}:
        val_loader = create_json_segment_dataloader(
            img_dir=val_img_dir,
            json_label_dir=val_json_dir,
            img_size=imgsz,
            batch_size=batch_size // WORLD_SIZE * 2,
            augment=False,
            workers=workers*2,
            shuffle=False,
            hyp=hyp
        )[0]

    # 模型属性配置
    model.num_classes = num_classes
    model.hyp = hyp
    model.names = names

    # 处理类别权重
    if opt.class_weights:
        # 解析用户提供的类别权重
        class_weights = parse_class_weights(opt.class_weights, num_classes)
        LOGGER.info(f"使用自定义类别权重: {class_weights.tolist()}")
    else:
        # 计算基于数据分布的权重
        class_weights = seg_labels_to_class_weights(dataset.json_files, num_classes)
        LOGGER.info(f"使用数据驱动的类别权重: {class_weights.tolist()}")
    
    class_weights = class_weights.to(device)
    model.class_weights = class_weights

    # 损失函数 - 传入类别权重
    criterion = SegmentationLoss(
        num_classes=num_classes, 
        label_smoothing=hyp['label_smoothing'],
        class_weights=class_weights
    )
    
    # 控制AMP启用状态
    amp_enabled = opt.amp and cuda
    scaler = None
    if amp_enabled:
        try:
            from torch.cuda import amp
            scaler = amp.GradScaler(enabled=amp_enabled)
            LOGGER.info("已启用自动混合精度训练(AMP)")
        except ImportError:
            LOGGER.warning("当前PyTorch版本不支持AMP，将禁用AMP")
            amp_enabled = False
            scaler = None
    else:
        LOGGER.info("已禁用自动混合精度训练(AMP)")

    stopper = EarlyStopping(patience=opt.patience)

    # 训练循环
    LOGGER.info(colorstr('开始训练（YOLOv8 Backbone语义分割版）!'))
    t0 = time.time()
    global_step = 0
    
    for epoch in range(start_epoch, epochs):
        model.train()
        mloss = torch.zeros(3, device=device)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 6) % ('Epoch', 'gpu_mem', '总损失', 'CE损失', 'Jaccard损失', 'img_size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=len(train_loader), bar_format=TQDM_BAR_FORMAT)
        
        for i, (imgs, targets, paths) in pbar:
            imgs = imgs.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).long()

            # 前向传播
            if amp_enabled:
                from torch.cuda import amp
                with amp.autocast():
                    pred = model(imgs)
                    loss, loss_items = criterion(pred, targets)
                    loss_items = torch.tensor(loss_items, device=device)
            else:
                pred = model(imgs)
                loss, loss_items = criterion(pred, targets)
                loss_items = torch.tensor(loss_items, device=device)

            # 反向传播与梯度更新
            if amp_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % accumulate == 0 or i == len(train_loader) - 1:
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # 日志更新
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = f'{torch.cuda.memory_reserved() / 1E9:.3g}G' if cuda else 'N/A'
                pbar.set_description(('%10s' * 2 + '%10.4g' * 4) %
                                    (f'{epoch}/{epochs - 1}', mem, *mloss, imgs.shape[-1]))

                # TensorBoard日志
                if global_step % 10 == 0:
                    tb_writer.add_scalar('Train/Total_Loss', loss_items[0], global_step)
                    tb_writer.add_scalar('Train/CE_Loss', loss_items[1], global_step)
                    tb_writer.add_scalar('Train/Jaccard_Loss', loss_items[2], global_step)
                    tb_writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)

                if global_step % 100 == 0 and global_step != 0:
                    num_samples = min(3, imgs.size(0))
                    for s in range(num_samples):
                        img_np = imgs[s].permute(1, 2, 0).cpu().detach().numpy()
                        target_np = targets[s].cpu().detach().numpy()
                        pred_np = torch.argmax(pred[s], dim=0).cpu().detach().numpy()

                        target_rgb = mask_to_rgb(target_np, CAMVID_COLORS)
                        pred_rgb = mask_to_rgb(pred_np, CAMVID_COLORS)
                        diff_rgb = visualize_prediction_difference(target_np, pred_np, CAMVID_COLORS)

                        tb_writer.add_image(f'Train/Sample_{s}/Input_Image', img_np, global_step, dataformats='HWC')
                        tb_writer.add_image(f'Train/Sample_{s}/Ground_Truth', target_rgb, global_step, dataformats='HWC')
                        tb_writer.add_image(f'Train/Sample_{s}/Prediction', pred_rgb, global_step, dataformats='HWC')
                        tb_writer.add_image(f'Train/Sample_{s}/Differences', diff_rgb, global_step, dataformats='HWC')

                global_step += 1

        # 学习率更新
        scheduler.step()

        # 验证与模型保存
        if RANK in {-1, 0}:
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

                # 验证日志
                tb_writer.add_scalar('Validation/mIoU', results[0], epoch)
                for cls_idx, cls_name in enumerate(names):
                    if cls_idx < len(maps):
                        tb_writer.add_scalar(f'Validation/Class_IoU/{cls_name}', maps[cls_idx], epoch)

                # 验证集可视化
                model.eval()
                with torch.no_grad():
                    val_batch = next(iter(val_loader))
                    val_imgs, val_targets, _ = val_batch
                    val_imgs = val_imgs.to(device)
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
                        
                        tb_writer.add_image(f'Validation/Sample_{s}/Input_Image', img_np, epoch, dataformats='HWC')
                        tb_writer.add_image(f'Validation/Sample_{s}/Ground_Truth', target_rgb, epoch, dataformats='HWC')
                        tb_writer.add_image(f'Validation/Sample_{s}/Prediction', pred_rgb, epoch, dataformats='HWC')
                        tb_writer.add_image(f'Validation/Sample_{s}/Differences', diff_rgb, epoch, dataformats='HWC')

            # 保存模型
            fi = fitness(np.array(results).reshape(1, -1))
            if fi > best_fitness:
                best_fitness = fi
            save = (not nosave) or (final_epoch and not opt.evolve)
            if save:
                torch.save({'model': ema.ema, 'optimizer': optimizer.state_dict(), 
                           'epoch': epoch, 'best_fitness': best_fitness}, last)
                if fi == best_fitness:
                    torch.save({'model': ema.ema}, best)
                    LOGGER.info(f"更新最佳模型 (mIoU: {results[0]:.4f})")

            # 早停检查
            if stopper(epoch=epoch, fitness=fi):
                break

    # 训练结束
    if RANK in {-1, 0}:
        if tb_writer:
            tb_writer.close()
            
        LOGGER.info(f'\n训练完成 ({(time.time() - t0) / 3600:.2f} 小时)')
        LOGGER.info(f"最佳模型保存至: {best}")
        strip_optimizer(best)

    return best_fitness


# -------------------------- 命令行参数与主函数 --------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='初始权重路径（YOLOv8预训练权重或自定义权重）')
    parser.add_argument('--cfg', type=str, default='yolov8.yaml', help='模型配置文件（YOLOv8语义分割版）')
    parser.add_argument('--data', type=str, default='/root/BestYOLO/CamVid/data.yaml', help='数据集配置文件（JSON版本）')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4, help='批量大小')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', default='', help='设备配置, 例如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='冻结层索引')
    parser.add_argument('--patience', type=int, default=150, help='早停 patience')
    parser.add_argument('--single-cls', action='store_true', help='单类别训练')
    parser.add_argument('--sync-bn', action='store_true', help='使用同步BN')
    parser.add_argument('--cos-lr', action='store_true', help='使用余弦学习率调度')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='恢复训练')
    parser.add_argument('--save-dir', type=str, default='runs/train-yolov8-seg', help='结果保存目录（YOLOv8专用）')
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
    parser.add_argument('--amp', action='store_true', default=False, 
                        help='启用自动混合精度训练（默认禁用）')
    parser.add_argument('--class-weights', type=str, default='weight.yaml', 
                        help='自定义类别权重，可以是逗号分隔的权重值或权重文件路径，例如 "1.0,2.0,3.0" 或 weights.yaml')
    
    opt = parser.parse_args()
    check_requirements(exclude=('tensorboard', 'thop'))
    
    # 设备选择
    device = select_device(opt.device, batch_size=opt.batch_size)
    if device.type == 'cpu':
        opt.sync_bn = False  # CPU不支持同步BN  
        opt.amp = False  # CPU上强制禁用AMP
    
    # 初始化训练
    callbacks = Callbacks()
    train(hyp=opt.hyp, opt=opt, device=device, callbacks=callbacks)
