import argparse
import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # 项目根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# 导入YOLOv8语义分割相关组件
from seg_jaccardloss_yolov8 import YOLOv8Seg, create_json_segment_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size,
                           check_requirements, check_yaml, colorstr, increment_path, print_args)
from utils.torch_utils import select_device, smart_inference_mode

# CamVid颜色映射表
CAMVID_COLORS = [
    [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
    [60, 40, 222], [128, 128, 0], [192, 128, 128], [64, 64, 128],
    [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 0]  # 最后一个是unlabelled（类别11，忽略）
]


# -------------------------- Jaccard损失计算类 --------------------------
class JaccardLoss:
    """专用Jaccard损失计算，针对语义分割任务优化"""
    def __init__(self, num_classes, ignore_index=11, smooth=1e-6):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def __call__(self, preds, targets):
        """
        计算Jaccard损失
        preds: 模型输出 (batch, num_classes, H, W)
        targets: 目标掩码 (batch, H, W)
        """
        # 过滤忽略类别
        mask = (targets != self.ignore_index)
        if not mask.any():
            return torch.tensor(0.0, device=preds.device)
            
        # 转换为One-Hot编码
        preds_softmax = F.softmax(preds, dim=1)
        one_hot_target = F.one_hot(targets.clamp(0, self.num_classes-1), self.num_classes)
        one_hot_target = one_hot_target.permute(0, 3, 1, 2).float()  # 转换为(batch, num_classes, H, W)
        
        # 应用掩码
        preds_softmax = preds_softmax * mask.unsqueeze(1).float()
        one_hot_target = one_hot_target * mask.unsqueeze(1).float()
        
        # 计算交并集
        intersection = torch.sum(preds_softmax * one_hot_target, dim=(1, 2, 3))
        union = torch.sum(preds_softmax + one_hot_target - preds_softmax * one_hot_target, dim=(1, 2, 3))
        
        # 计算Jaccard指数和损失
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        return 1 - torch.mean(jaccard)


# -------------------------- 混淆矩阵类 --------------------------
class SegmentationConfusionMatrix:
    """语义分割专用混淆矩阵，处理类别索引（忽略unlabelled类别）"""
    def __init__(self, num_classes, ignore_index=11):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def process_batch(self, preds, targets):
        """处理单个批次的预测和目标掩码"""
        # 展平为一维数组
        preds = preds.flatten()
        targets = targets.flatten()

        # 过滤忽略的类别（unlabelled，索引11）
        mask = (targets != self.ignore_index)
        preds = preds[mask]
        targets = targets[mask]

        # 更新混淆矩阵
        for t, p in zip(targets, preds):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.matrix[t, p] += 1

    def compute_iou(self):
        """计算每个类别的IoU和平均mIoU"""
        ious = []
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
                
            true_positive = self.matrix[cls, cls]
            false_positive = self.matrix[:, cls].sum() - true_positive
            false_negative = self.matrix[cls, :].sum() - true_positive
            union = true_positive + false_positive + false_negative
            
            ious.append(true_positive / union if union != 0 else 0.0)
        
        return np.mean(ious), ious

    def plot(self, save_dir, names):
        """可视化混淆矩阵"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(self.matrix, cmap='Blues')
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                plt.text(j, i, self.matrix[i, j], 
                         horizontalalignment='center', 
                         verticalalignment='center',
                         color='white' if self.matrix[i, j] > self.matrix.max() / 2 else 'black')
        
        plt.xticks(range(self.num_classes), names, rotation=45)
        plt.yticks(range(self.num_classes), names)
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.title('YOLOv8 Segmentation Confusion Matrix (Jaccard)')
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png')
        plt.close()


# -------------------------- 结果可视化函数 --------------------------
def visualize_results(img, pred_mask, true_mask, save_path, names):
    """可视化原图、真实掩码和预测掩码"""
    img_np = img.permute(1, 2, 0).cpu().numpy() * 255
    img_np = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    pred_colored = np.zeros_like(img_np)
    true_colored = np.zeros_like(img_np)
    
    for cls in range(len(CAMVID_COLORS)):
        pred_colored[pred_mask == cls] = CAMVID_COLORS[cls]
        true_colored[true_mask == cls] = CAMVID_COLORS[cls]
    
    pred_overlay = cv2.addWeighted(img_np, 0.5, pred_colored, 0.5, 0)
    true_overlay = cv2.addWeighted(img_np, 0.5, true_colored, 0.5, 0)
    
    h, w = img_np.shape[:2]
    title = np.ones((50, w*3, 3), dtype=np.uint8) * 255
    cv2.putText(title, '原图', (w//3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(title, '真实掩码', (w + w//3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(title, '预测掩码', (2*w + w//3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    
    combined = np.vstack((title, np.hstack((img_np, true_overlay, pred_overlay))))
    cv2.imwrite(str(save_path), combined)


# -------------------------- 核心验证函数 --------------------------
@smart_inference_mode()
def run(
        data,
        weights=None,
        batch_size=8,
        imgsz=640,
        device='',
        workers=4,
        single_cls=False,
        augment=False,
        verbose=False,
        project=ROOT / 'runs/val-yolov8-seg-jaccard',
        name='exp',
        exist_ok=False,
        half=True,
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
):
    # 初始化模式
    training = model is not None
    total_jaccard_loss = 0.0  # 累积Jaccard损失
    
    if training:
        # 训练中调用
        device = next(model.parameters()).device
        nc = model.num_classes
        names = model.names
        
        # 初始化Jaccard损失计算器
        jaccard_loss_fn = JaccardLoss(num_classes=nc)
        
        # 模型数据类型调整
        if half:
            model = model.half()
        else:
            model = model.float()
    else:
        # 独立运行模式
        device = select_device(device, batch_size=batch_size)
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        (save_dir / 'visualizations').mkdir(parents=True, exist_ok=True)
        
        # 加载数据集配置
        data_dict = check_dataset(data)
        nc = 1 if single_cls else int(data_dict['nc'])
        
        # 初始化Jaccard损失计算器
        jaccard_loss_fn = JaccardLoss(num_classes=nc)
        
        # 加载YOLOv8Seg模型
        model = YOLOv8Seg(cfg='yolov8.yaml', num_classes=nc).to(device)
        
        # 加载模型权重
        if weights:
            ckpt = torch.load(weights, map_location=device)
            if 'model' in ckpt:
                ckpt = ckpt['model']
            model.load_state_dict(ckpt.float().state_dict())
        model.eval()
        
        # 检查图像尺寸
        stride = model.stride
        imgsz = check_img_size(imgsz, s=stride)
        
        # 类别名称
        names = data_dict['names'] if not single_cls else ['class0']
        
        # 创建数据加载器
        dataloader = create_json_segment_dataloader(
            img_dir=data_dict['val_img'],
            json_label_dir=data_dict['val_json'],
            img_size=imgsz,
            batch_size=batch_size,
            augment=False,
            workers=workers,
            shuffle=False
        )[0]

    # 初始化评估工具
    confusion_matrix = SegmentationConfusionMatrix(nc, ignore_index=11)
    seen = 0
    dt = Profile(), Profile(), Profile()

    # 开始验证
    model.eval()
    pbar = tqdm(dataloader, desc=colorstr('val-jaccard: '), bar_format=TQDM_BAR_FORMAT)
    for batch_i, (imgs, masks, paths) in enumerate(pbar):
        with dt[0]:
            # 数据预处理
            if imgs.dtype == torch.uint8 or float(imgs.max()) > 1.0:
                imgs = imgs.to(device, non_blocking=True).float() / 255.0
            else:
                imgs = imgs.to(device, non_blocking=True).float()

            if half:
                imgs = imgs.half()
            masks = masks.to(device, non_blocking=True).long()
            nb, _, height, width = imgs.shape

        with dt[1]:
            # 模型推理
            preds = model(imgs, augment=augment) if not training else model(imgs)
            pred_masks = torch.argmax(preds, dim=1)
            
            # 计算Jaccard损失
            batch_loss = jaccard_loss_fn(preds, masks)
            total_jaccard_loss += batch_loss.item() * nb  # 累积批次损失

        with dt[2]:
            # 后处理
            for si in range(nb):
                seen += 1
                path = Path(paths[si])
                true_mask = masks[si].cpu().numpy()
                pred_mask = pred_masks[si].cpu().numpy()

                # 更新混淆矩阵
                confusion_matrix.process_batch(pred_mask, true_mask)

                # 可视化结果
                if plots and batch_i < 3 and si < 3:
                    visualize_path = save_dir / 'visualizations' / f'{path.stem}_val_jaccard.png'
                    visualize_results(imgs[si], pred_mask, true_mask, visualize_path, names)

        # 进度条更新
        pbar.set_postfix({
            '样本数': seen,
            '平均Jaccard损失': total_jaccard_loss / seen if seen > 0 else 0
        })

    # 计算最终评估指标
    final_miou, per_cls_iou = confusion_matrix.compute_iou()
    avg_jaccard_loss = total_jaccard_loss / seen if seen > 0 else 0.0

    # 打印评估结果
    LOGGER.info('\n' + '='*60)
    LOGGER.info(f"YOLOv8语义分割验证结果 (Jaccard) | 验证集总数: {seen} 张图像")
    LOGGER.info(f"整体mIoU: {final_miou:.4f}")
    LOGGER.info(f"平均Jaccard损失: {avg_jaccard_loss:.4f}")  # 重点展示Jaccard损失
    LOGGER.info('-'*60)
    LOGGER.info(f"{'类别名称':<15} {'IoU值':<10}")
    LOGGER.info('-'*25)
    cls_idx = 0
    for cls in range(nc):
        if cls != 11:
            LOGGER.info(f"{names[cls]:<15} {per_cls_iou[cls_idx]:.4f}")
            cls_idx += 1
    LOGGER.info('='*60)

    # 保存混淆矩阵图
    if plots and not training:
        confusion_matrix.plot(save_dir, names)

    # 打印推理速度
    t = tuple(x.t / seen * 1000 for x in dt)
    LOGGER.info(f"推理速度: 预处理 {t[0]:.1f}ms/张, 推理 {t[1]:.1f}ms/张, 后处理 {t[2]:.1f}ms/张")

    # 打印结果保存路径
    if not training:
        LOGGER.info(f"验证结果保存至: {colorstr('bold', save_dir)}")

    # 返回评估指标（扩展为4个指标，匹配fitness函数要求）
    # 补充两个0.0作为占位符，确保与训练代码兼容
    return (final_miou, avg_jaccard_loss, 0.0, 0.0), per_cls_iou, t


# -------------------------- 命令行参数解析 --------------------------
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'CamVid/data.yaml', 
                      help='数据集配置文件')
    parser.add_argument('--weights', type=str, 
                      default='runs/train-yolov8-seg/weights/best.pt',
                      help='YOLOv8语义分割模型权重路径')
    parser.add_argument('--batch-size', type=int, default=8, help='批量大小')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, 
                      help='输入图像尺寸')
    parser.add_argument('--device', default='', help='设备配置，例如 0 或 cpu')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--single-cls', action='store_true', help='单类别数据集')
    parser.add_argument('--augment', action='store_true', help='增强推理')
    parser.add_argument('--verbose', action='store_true', help='详细输出每个类别的IoU')
    parser.add_argument('--project', default=ROOT / 'runs/val-yolov8-seg-jaccard', 
                      help='Jaccard验证结果保存路径')
    parser.add_argument('--name', default='exp', help='实验名称')
    parser.add_argument('--exist-ok', action='store_true', help='允许覆盖现有实验目录')
    parser.add_argument('--half', action='store_true', help='使用FP16半精度推理')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    print_args(vars(opt))
    return opt


# -------------------------- 主函数 --------------------------
def main(opt):
    check_requirements()
    run(** vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
