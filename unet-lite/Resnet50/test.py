import argparse
import os
import sys
import time
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 路径配置
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# 创建可视化结果保存目录
VISUALIZATION_DIR = Path("visualization_results")
VISUALIZATION_DIR.mkdir(exist_ok=True, parents=True)

# CamVid颜色映射表(RGB)用于可视化
CAMVID_COLORS = [
    [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
    [60, 40, 222], [128, 128, 0], [192, 128, 128], [64, 64, 128],
    [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 0]  # 最后一个是unlabelled
]
CAMVID_COLORS_NP = np.array(CAMVID_COLORS, dtype=np.uint8)


# -------------------------- 工具函数 --------------------------
def mask_to_rgb(mask, color_map=CAMVID_COLORS_NP):
    """将类别索引掩码转换为RGB彩色掩码"""
    return color_map[mask]


def visualize_results(images, ground_truths, predictions, paths, class_names, 
                     num_samples=5, save=True, show=False):
    """
    可视化原图、真实标签和预测结果
    
    参数:
        images: 原始图像张量 (batch, 3, h, w)
        ground_truths: 真实标签张量 (batch, h, w)
        predictions: 预测结果张量 (batch, h, w)
        paths: 图像路径列表
        class_names: 类别名称列表
        num_samples: 要可视化的样本数量
        save: 是否保存可视化结果
        show: 是否显示可视化结果
    """
    # 确保不超过批次大小
    num_samples = min(num_samples, len(images))
    
    # 随机选择样本
    indices = random.sample(range(len(images)), num_samples)
    
    for i in indices:
        # 转换为numpy数组并调整维度
        img = images[i].permute(1, 2, 0).cpu().numpy()  # (h, w, 3)
        gt = ground_truths[i].cpu().numpy()             # (h, w)
        pred = predictions[i].cpu().numpy()             # (h, w)
        
        # 转换掩码为RGB
        gt_rgb = mask_to_rgb(gt)
        pred_rgb = mask_to_rgb(pred)
        
        # 创建图像标题（使用文件名）
        img_name = Path(paths[i]).name
        
        # 创建画布
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"图像: {img_name}", fontsize=16)
        
        # 显示原图
        axes[0].imshow(img)
        axes[0].set_title("原图")
        axes[0].axis("off")
        
        # 显示真实标签
        axes[1].imshow(gt_rgb)
        axes[1].set_title("真实标签")
        axes[1].axis("off")
        
        # 显示预测结果
        axes[2].imshow(pred_rgb)
        axes[2].set_title("预测结果")
        axes[2].axis("off")
        
        # 添加类别图例
        create_legend(fig, class_names)
        
        plt.tight_layout()
        
        # 保存图像
        if save:
            save_path = VISUALIZATION_DIR / f"result_{img_name.split('.')[0]}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"可视化结果已保存至: {save_path}")
        
        # 显示图像
        if show:
            plt.show()
        else:
            plt.close(fig)


def create_legend(fig, class_names):
    """创建类别颜色图例"""
    handles = []
    for i, class_name in enumerate(class_names):
        color = CAMVID_COLORS_NP[i] / 255.0  # 转换为0-1范围
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color))
    
    # 在图像右侧添加图例
    fig.legend(handles, class_names, loc='center right', 
              bbox_to_anchor=(1.15, 0.5), fontsize=8)


# -------------------------- 模型定义 --------------------------
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
    """ResNet50主干网络"""
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
        self.feat_channels = [256, 512, 1024]

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
        
        # 四个阶段的特征
        f1 = self.layer1(x)  # 256通道
        f2 = self.layer2(f1) # 512通道
        f3 = self.layer3(f2) # 1024通道
        
        return [f1, f2, f3]


class SegmentHead(nn.Module):
    """语义分割头，处理多尺度特征"""
    def __init__(self, num_classes: int = 12, in_channels: List[int] = [256, 512, 1024]):
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


class ResNet50Seg(nn.Module):
    """完整的ResNet50语义分割模型"""
    def __init__(self, cfg: Dict, num_classes: Optional[int] = None):
        super().__init__()
        # 解析配置文件
        if isinstance(cfg, str):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播入口"""
        return self._forward_once(x)

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
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


# -------------------------- 测试数据集类 --------------------------
class TestSegmentDataset(Dataset):
    """语义分割测试数据集加载类"""
    def __init__(self, img_dir, label_dir, img_size=640):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        
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
        
        print(f"加载测试数据集: {len(self.img_files)} 张图像，{len(self.label_files)} 个掩码")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # 加载图像和掩码
        img_path = self.img_files[index]
        label_path = self.label_files[index]
        
        # 读取图像（RGB）和掩码（单通道，类别索引）
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # 转为单通道灰度图（像素值为类别索引）
        
        # 调整尺寸并填充
        img, label = self._resize_and_pad(img, label)
        
        # 转换为张量
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # 归一化到[0,1]
        label = torch.from_numpy(np.array(label)).long()  # 掩码保持整数类别
        
        return img, label, str(img_path)

    def _resize_and_pad(self, img, label):
        """调整图像和掩码尺寸并填充至目标大小"""
        w, h = img.size
        scale = min(self.img_size / w, self.img_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 缩放（图像用双线性，掩码用最近邻）
        img = img.resize((new_w, new_h), Image.BILINEAR)
        label = label.resize((new_w, new_h), Image.NEAREST)
        
        # 计算填充
        pad_w = self.img_size - new_w
        pad_h = self.img_size - new_h
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        
        # 填充图像和掩码
        new_img = Image.new('RGB', (self.img_size, self.img_size), (128, 128, 128))  # 灰色填充
        new_label = Image.new('L', (self.img_size, self.img_size), 0)  # 背景填充为0类
        
        new_img.paste(img, (pad_left, pad_top))
        new_label.paste(label, (pad_left, pad_top))
        
        return new_img, new_label


# -------------------------- 评估指标计算 --------------------------
class SegmentationMetrics:
    """语义分割评估指标计算"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    def update(self, pred, target):
        """更新混淆矩阵"""
        # 确保预测和目标尺寸一致
        if pred.shape != target.shape:
            pred = F.interpolate(
                pred.unsqueeze(0), 
                size=target.shape[1:], 
                mode='nearest'
            ).squeeze(0)
        
        # 转换为numpy数组
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        
        # 计算混淆矩阵
        mask = (target >= 0) & (target < self.num_classes)
        label = self.num_classes * target[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=self.num_classes **2)
        self.confusion_matrix += count.reshape(self.num_classes, self.num_classes)
    
    def get_metrics(self):
        """计算并返回各项指标"""
        # 计算每个类别的IoU
        iou = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + 
            np.sum(self.confusion_matrix, axis=0) - 
            np.diag(self.confusion_matrix) + 
            1e-10  # 防止除零
        )
        
        # 计算mIoU
        miou = np.nanmean(iou)
        
        # 计算准确率
        accuracy = np.diag(self.confusion_matrix).sum() / (
            self.confusion_matrix.sum() + 1e-10
        )
        
        # 计算每个类别的准确率
        class_acc = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + 1e-10
        )
        
        return {
            'mIoU': miou,
            'IoU': iou,
            'Accuracy': accuracy,
            'Class_Accuracy': class_acc
        }


# -------------------------- 测试主函数 --------------------------
def test(opt):
    """测试模型主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据配置
    with open(opt.data, 'r') as f:
        data_dict = yaml.safe_load(f)
    
    # 获取测试集路径
    test_img_dir = data_dict.get('test_img', data_dict.get('test', ''))
    test_label_dir = data_dict.get('test_label', os.path.join(data_dict.get('test', ''), 'labels'))
    num_classes = int(data_dict['nc'])
    class_names = data_dict['names']
    
    # 验证测试集路径
    if not os.path.exists(test_img_dir):
        raise FileNotFoundError(f"测试图像路径不存在: {test_img_dir}")
    if not os.path.exists(test_label_dir):
        raise FileNotFoundError(f"测试掩码路径不存在: {test_label_dir}")
    
    # 创建测试集加载器
    test_dataset = TestSegmentDataset(
        img_dir=test_img_dir,
        label_dir=test_label_dir,
        img_size=opt.imgsz
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=True
    )
    
    # 加载模型
    print(f"加载模型权重: {opt.weights}")
    model = ResNet50Seg(opt.cfg, num_classes=num_classes).to(device)
    checkpoint = torch.load(opt.weights, map_location=device)
    
    # 兼容不同的权重存储格式
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'].state_dict() if hasattr(checkpoint['model'], 'state_dict') else checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 打印模型参数信息
    print("\n模型参数信息:")
    print(f"类别数量: {num_classes}")
    print(f"输入图像尺寸: {opt.imgsz}x{opt.imgsz}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 初始化评估指标
    metrics = SegmentationMetrics(num_classes)
    
    # 用于可视化的变量
    visualize_samples = []
    
    # 开始测试
    print("\n开始测试...")
    start_time = time.time()
    with torch.no_grad():
        for imgs, targets, paths in tqdm(test_loader, desc="测试进度"):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # 模型推理
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            
            # 更新评估指标
            for pred, target in zip(preds, targets):
                metrics.update(pred, target)
            
            # 收集用于可视化的样本
            if len(visualize_samples) < opt.visualize_num:
                # 限制收集的样本数量
                remaining = opt.visualize_num - len(visualize_samples)
                take = min(remaining, len(imgs))
                visualize_samples.extend(zip(
                    imgs[:take], 
                    targets[:take], 
                    preds[:take],
                    paths[:take]
                ))
    
    # 执行可视化
    if opt.visualize and visualize_samples:
        print("\n生成可视化结果...")
        # 分离可视化样本的各个部分
        vis_imgs = torch.stack([x[0] for x in visualize_samples])
        vis_targets = torch.stack([x[1] for x in visualize_samples])
        vis_preds = torch.stack([x[2] for x in visualize_samples])
        vis_paths = [x[3] for x in visualize_samples]
        
        # 生成可视化结果
        visualize_results(
            vis_imgs, 
            vis_targets, 
            vis_preds, 
            vis_paths,
            class_names,
            num_samples=opt.visualize_num,
            save=opt.save_visualization,
            show=opt.show_visualization
        )
    
    # 计算并打印结果
    test_time = time.time() - start_time
    print(f"\n测试完成，耗时: {test_time:.2f}秒")
    print(f"测试样本数量: {len(test_dataset)}")
    print(f"平均每张图像处理时间: {test_time / len(test_dataset):.4f}秒")
    
    # 获取评估指标
    results = metrics.get_metrics()
    
    # 打印详细指标
    print("\n===== 测试集评估结果 =====")
    print(f"mIoU: {results['mIoU']:.4f}")
    print(f"总体准确率: {results['Accuracy']:.4f}")
    
    print("\n每个类别的IoU:")
    for i, (class_name, iou) in enumerate(zip(class_names, results['IoU'])):
        print(f"  类别 {i} ({class_name}): {iou:.4f}")
    
    print("\n每个类别的准确率:")
    for i, (class_name, acc) in enumerate(zip(class_names, results['Class_Accuracy'])):
        print(f"  类别 {i} ({class_name}): {acc:.4f}")


if __name__ == '__main__':
    # 命令行参数
    parser = argparse.ArgumentParser(description='语义分割模型测试脚本')
    parser.add_argument('--weights', type=str, default='/root/BestYOLO/Resnet50/runs/train-diceloss/weights/best.pt', help='模型权重文件路径')
    parser.add_argument('--cfg', type=str, default='resnet50.yaml', help='模型配置文件')
    parser.add_argument('--data', type=str, default='/root/BestYOLO/CamVid/data.yaml', help='数据集配置文件')
    parser.add_argument('--batch-size', type=int, default=16, help='批量大小')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', default='', help='设备配置, 例如 cuda 或 cpu')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--visualize', action='store_true', default=True, help='是否进行可视化')
    parser.add_argument('--visualize-num', type=int, default=5, help='可视化样本数量')
    parser.add_argument('--save-visualization', action='store_true', default=True, help='是否保存可视化结果')
    parser.add_argument('--show-visualization', action='store_true', default=False, help='是否显示可视化结果')
    
    opt = parser.parse_args()
    test(opt)
    