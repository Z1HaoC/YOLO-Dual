import argparse
import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
UTILS_ROOT = "/root/BestYOLO"
if UTILS_ROOT not in sys.path:
    sys.path.insert(0, UTILS_ROOT) 


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # 项目根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# 导入训练代码相关组件（关键：改为导入ResNet50Seg，而非YOLOv8Seg）
from seg_diceloss_Resnet50 import ResNet50Seg, create_json_segment_dataloader  # 重点修改：导入ResNet50Seg
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size,
                           check_requirements, check_yaml, colorstr, increment_path, print_args)
from utils.torch_utils import select_device, smart_inference_mode

# CamVid颜色映射表（与训练代码保持一致）
CAMVID_COLORS = [
    [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
    [60, 40, 222], [128, 128, 0], [192, 128, 128], [64, 64, 128],
    [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 0]  # 最后一个是unlabelled（类别11，忽略）
]


# -------------------------- Dice损失计算类（保持不变） --------------------------
class DiceLoss:
    """专用Dice损失计算，针对语义分割任务优化"""
    def __init__(self, num_classes, ignore_index=11, smooth=1e-6, class_weights=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.class_weights = class_weights if class_weights is not None else torch.ones(num_classes)

    def __call__(self, preds, targets):
        mask = (targets != self.ignore_index)
        if not mask.any():
            return torch.tensor(0.0, device=preds.device)
            
        preds_softmax = F.softmax(preds, dim=1)
        one_hot_target = F.one_hot(targets.clamp(0, self.num_classes-1), self.num_classes)
        one_hot_target = one_hot_target.permute(0, 3, 1, 2).float()
        
        preds_softmax = preds_softmax * mask.unsqueeze(1).float()
        one_hot_target = one_hot_target * mask.unsqueeze(1).float()
        
        if self.class_weights.device != preds.device:
            self.class_weights = self.class_weights.to(preds.device)
        
        weighted_preds = preds_softmax * self.class_weights.view(1, -1, 1, 1)
        intersection = torch.sum(weighted_preds * one_hot_target, dim=(1, 2, 3))
        pred_sum = torch.sum(weighted_preds, dim=(1, 2, 3))
        target_sum = torch.sum(one_hot_target, dim=(1, 2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        return 1.0 - torch.mean(dice)


# -------------------------- 混淆矩阵类（保持不变） --------------------------
class SegmentationConfusionMatrix:
    """语义分割专用混淆矩阵，处理类别索引（忽略unlabelled类别）"""
    def __init__(self, num_classes, ignore_index=11):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def process_batch(self, preds, targets):
        preds = preds.flatten()
        targets = targets.flatten()
        mask = (targets != self.ignore_index)
        preds = preds[mask]
        targets = targets[mask]

        for t, p in zip(targets, preds):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.matrix[t, p] += 1

    def compute_iou(self):
        ious = []
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            tp = self.matrix[cls, cls]
            fp = self.matrix[:, cls].sum() - tp
            fn = self.matrix[cls, :].sum() - tp
            union = tp + fp + fn
            ious.append(tp / union if union != 0 else 0.0)
        return np.mean(ious), ious

    def plot(self, save_dir, names):
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
        plt.title('ResNet50 Segmentation Confusion Matrix (Dice)')  # 修改：匹配ResNet50模型
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png')
        plt.close()


# -------------------------- 结果可视化函数（保持不变） --------------------------
def visualize_results(img, pred_mask, true_mask, save_path, names):
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


# -------------------------- 核心验证函数（关键修改） --------------------------
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
        project=ROOT / 'runs/val-resnet50-seg-dice',  # 修改：匹配ResNet50保存路径
        name='exp',
        exist_ok=False,
        half=True,
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        class_weights=None,
):
    training = model is not None
    total_dice_loss = 0.0
    
    if training:
        # 训练中调用：直接使用传入的ResNet50Seg模型（无需修改）
        device = next(model.parameters()).device
        nc = model.num_classes
        names = model.names
        class_weights = getattr(model, 'class_weights', None)
        dice_loss_fn = DiceLoss(num_classes=nc, class_weights=class_weights)
        
        if half:
            model = model.half()
        else:
            model = model.float()
    else:
        # 独立运行模式：加载ResNet50Seg模型（重点修改：替换YOLOv8Seg为ResNet50Seg）
        device = select_device(device, batch_size=batch_size)
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        (save_dir / 'visualizations').mkdir(parents=True, exist_ok=True)
        
        # 加载数据集配置（补充容错：适配训练代码的data.yaml路径）
        data_dict = check_dataset(data)
        nc = 1 if single_cls else int(data_dict['nc'])
        dice_loss_fn = DiceLoss(num_classes=nc, class_weights=class_weights)
        
        # 关键修改：加载ResNet50Seg模型（而非YOLOv8Seg）
        model = ResNet50Seg(cfg='resnet50.yaml', num_classes=nc).to(device)  # 与训练代码的模型配置一致
        
        # 加载权重（逻辑不变，适配ResNet50Seg的权重格式）
        if weights:
            ckpt = torch.load(weights, map_location=device)
            csd = ckpt['model'].float().state_dict() if 'model' in ckpt else ckpt.float().state_dict()
            csd = {k: v for k, v in csd.items() if k in model.state_dict()}  # 过滤不匹配的权重
            model.load_state_dict(csd, strict=False)
            LOGGER.info(f"加载ResNet50Seg权重: {len(csd)}/{len(model.state_dict())} 项匹配")
        model.eval()
        
        # 检查图像尺寸（使用ResNet50的stride）
        stride = model.stride
        imgsz = check_img_size(imgsz, s=stride.max())
        
        # 类别名称与数据加载器（与训练代码保持一致）
        names = data_dict['names'] if not single_cls else ['class0']
        dataloader = create_json_segment_dataloader(
            img_dir=data_dict.get('val_img', data_dict.get('val', '')),  # 适配训练代码的路径键
            json_label_dir=data_dict.get('val_json', os.path.join(data_dict.get('val_img', ''), 'json')),  # 容错：自动补全JSON路径
            img_size=imgsz,
            batch_size=batch_size,
            augment=False,
            workers=workers,
            shuffle=False
        )[0]

    # 初始化评估工具（逻辑不变）
    confusion_matrix = SegmentationConfusionMatrix(nc, ignore_index=11)
    seen = 0
    dt = Profile(), Profile(), Profile()

    # 开始验证（逻辑不变）
    model.eval()
    pbar = tqdm(dataloader, desc=colorstr('val-dice: '), bar_format=TQDM_BAR_FORMAT)
    for batch_i, (imgs, masks, paths) in enumerate(pbar):
        with dt[0]:
            if imgs.dtype == torch.uint8 or float(imgs.max()) > 1.0:
                imgs = imgs.to(device, non_blocking=True).float() / 255.0
            else:
                imgs = imgs.to(device, non_blocking=True).float()
            if half:
                imgs = imgs.half()
            masks = masks.to(device, non_blocking=True).long()
            nb, _, height, width = imgs.shape

        with dt[1]:
            preds = model(imgs, augment=augment) if not training else model(imgs)
            pred_masks = torch.argmax(preds, dim=1)
            batch_loss = dice_loss_fn(preds, masks)
            total_dice_loss += batch_loss.item() * nb

        with dt[2]:
            for si in range(nb):
                seen += 1
                path = Path(paths[si])
                true_mask = masks[si].cpu().numpy()
                pred_mask = pred_masks[si].cpu().numpy()
                confusion_matrix.process_batch(pred_mask, true_mask)
                
                # 可视化前3个批次的前3个样本
                if plots and batch_i < 3 and si < 3:
                    visualize_path = save_dir / 'visualizations' / f'{path.stem}_val_dice.png'
                    visualize_results(imgs[si], pred_mask, true_mask, visualize_path, names)

        pbar.set_postfix({
            '样本数': seen,
            '平均Dice损失': total_dice_loss / seen if seen > 0 else 0
        })

    # 计算与打印指标（逻辑不变）
    final_miou, per_cls_iou = confusion_matrix.compute_iou()
    avg_dice_loss = total_dice_loss / seen if seen > 0 else 0.0

    LOGGER.info('\n' + '='*60)
    LOGGER.info(f"ResNet50语义分割验证结果 (Dice) | 验证集总数: {seen} 张图像")  # 修改：匹配ResNet50
    LOGGER.info(f"整体mIoU: {final_miou:.4f}")
    LOGGER.info(f"平均Dice损失: {avg_dice_loss:.4f}")
    LOGGER.info('-'*60)
    LOGGER.info(f"{'类别名称':<15} {'IoU值':<10}")
    LOGGER.info('-'*25)
    cls_idx = 0
    for cls in range(nc):
        if cls != 11:
            LOGGER.info(f"{names[cls]:<15} {per_cls_iou[cls_idx]:.4f}")
            cls_idx += 1
    LOGGER.info('='*60)

    if plots and not training:
        confusion_matrix.plot(save_dir, names)

    t = tuple(x.t / seen * 1000 for x in dt)
    LOGGER.info(f"推理速度: 预处理 {t[0]:.1f}ms/张, 推理 {t[1]:.1f}ms/张, 后处理 {t[2]:.1f}ms/张")

    if not training:
        LOGGER.info(f"验证结果保存至: {colorstr('bold', save_dir)}")

    # 返回4个指标（适配训练代码的fitness函数要求）
    return (final_miou, avg_dice_loss, 0.0, 0.0), per_cls_iou, t


# -------------------------- 命令行参数解析（适配ResNet50） --------------------------
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'CamVid/data.yaml', 
                      help='数据集配置文件（与训练代码一致）')
    parser.add_argument('--weights', type=str, 
                      default='runs/train-resnet50-seg/weights/best.pt',  # 修改：匹配ResNet50训练权重路径
                      help='ResNet50语义分割模型权重路径')
    parser.add_argument('--batch-size', type=int, default=8, help='批量大小')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, 
                      help='输入图像尺寸（与训练代码一致）')
    parser.add_argument('--device', default='', help='设备配置，例如 0 或 cpu')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--single-cls', action='store_true', help='单类别数据集（CamVid无需启用）')
    parser.add_argument('--augment', action='store_true', help='增强推理')
    parser.add_argument('--verbose', action='store_true', help='详细输出每个类别的IoU')
    parser.add_argument('--project', default=ROOT / 'runs/val-resnet50-seg-dice', 
                      help='验证结果保存路径（与训练代码路径对应）')
    parser.add_argument('--name', default='exp', help='实验名称')
    parser.add_argument('--exist-ok', action='store_true', help='允许覆盖现有实验目录')
    parser.add_argument('--half', action='store_true', help='使用FP16半精度推理（GPU推荐启用）')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    print_args(vars(opt))
    return opt


# -------------------------- 主函数（保持不变） --------------------------
def main(opt):
    check_requirements()
    run(** vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)