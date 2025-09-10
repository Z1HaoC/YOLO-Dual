import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import cv2
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # 项目根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# 从YOLOv5语义分割训练代码导入相关组件（替换VGG16组件）
from seg import YOLOv5Seg, create_json_segment_dataloader  # 核心修改：使用YOLOv5Seg和JSON数据加载器

from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size,
                           check_requirements, check_yaml, colorstr, increment_path, print_args)
from utils.torch_utils import select_device, smart_inference_mode

# CamVid颜色映射表（与VGG16版本完全一致，确保可视化颜色统一）
CAMVID_COLORS = [
    [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
    [60, 40, 222], [128, 128, 0], [192, 128, 128], [64, 64, 128],
    [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 0]  # 最后一个是unlabelled（类别11，忽略）
]


# -------------------------- 混淆矩阵类（与VGG16版本完全一致，确保评估逻辑统一） --------------------------
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

        # 更新混淆矩阵（确保类别索引在有效范围内）
        for t, p in zip(targets, preds):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.matrix[t, p] += 1

    def compute_iou(self):
        """计算每个类别的IoU和平均mIoU（与VGG16版本完全一致）"""
        ious = []
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue  # 跳过忽略类别
                
            # 计算交并比：IoU = TP / (TP + FP + FN)
            true_positive = self.matrix[cls, cls]  # 真正例
            false_positive = self.matrix[:, cls].sum() - true_positive  # 假正例
            false_negative = self.matrix[cls, :].sum() - true_positive  # 假负例
            union = true_positive + false_positive + false_negative  # 并集
            
            ious.append(true_positive / union if union != 0 else 0.0)
        
        return np.mean(ious), ious  # 返回mIoU和各类别IoU

    def plot(self, save_dir, names):
        """可视化混淆矩阵（与VGG16版本完全一致，确保图表格式统一）"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(self.matrix, cmap='Blues')
        
        # 标注混淆矩阵数值（根据数值大小调整文字颜色）
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                plt.text(j, i, self.matrix[i, j], 
                         horizontalalignment='center', 
                         verticalalignment='center',
                         color='white' if self.matrix[i, j] > self.matrix.max() / 2 else 'black')
        
        # 设置坐标轴与标题
        plt.xticks(range(self.num_classes), names, rotation=45)
        plt.yticks(range(self.num_classes), names)
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.title('YOLOv5 Segmentation Confusion Matrix')  # 仅修改标题标识
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png')
        plt.close()


# -------------------------- 结果可视化函数（与VGG16版本完全一致，确保可视化效果统一） --------------------------
def visualize_results(img, pred_mask, true_mask, save_path, names):
    """可视化原图、真实掩码和预测掩码（格式与VGG16版本完全一致）"""
    # 转换图像格式（tensor -> numpy，反归一化）
    img_np = img.permute(1, 2, 0).cpu().numpy() * 255  # 训练时图像已归一化到[0,1]，此处反归一化
    img_np = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)  # 适配OpenCV的BGR格式
    
    # 为预测掩码和真实掩码上色（使用统一的CAMVID_COLORS）
    pred_colored = np.zeros_like(img_np)
    true_colored = np.zeros_like(img_np)
    
    for cls in range(len(CAMVID_COLORS)):
        pred_colored[pred_mask == cls] = CAMVID_COLORS[cls]
        true_colored[true_mask == cls] = CAMVID_COLORS[cls]
    
    # 叠加掩码到原图（透明度0.5，确保对比清晰）
    pred_overlay = cv2.addWeighted(img_np, 0.5, pred_colored, 0.5, 0)
    true_overlay = cv2.addWeighted(img_np, 0.5, true_colored, 0.5, 0)
    
    # 创建标题栏（区分三列图像）
    h, w = img_np.shape[:2]
    title = np.ones((50, w*3, 3), dtype=np.uint8) * 255  # 白色背景标题栏
    cv2.putText(title, '原图', (w//3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(title, '真实掩码', (w + w//3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(title, '预测掩码', (2*w + w//3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    
    # 拼接标题栏与图像（垂直堆叠标题，水平堆叠三列图像）
    combined = np.vstack((title, np.hstack((img_np, true_overlay, pred_overlay))))
    cv2.imwrite(str(save_path), combined)


# -------------------------- 核心验证函数（仅修改模型相关部分，其他逻辑与VGG16版本完全一致） --------------------------
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
        project=ROOT / 'runs/val-yolov5-seg',  # 核心修改：YOLOv5专用保存路径（区分VGG16）
        name='exp',
        exist_ok=False,
        half=True,
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
):
    # 初始化模式（训练中调用：复用模型/dataloader；独立运行：加载模型/dataloader）
    training = model is not None
    if training:
        # 训练过程中调用（复用YOLOv5Seg模型和数据加载器）
        device = next(model.parameters()).device
        nc = model.num_classes
        names = model.names
        
        # 确保模型数据类型与输入匹配（半精度/单精度）
        if half:
            model = model.half()
        else:
            model = model.float()
    else:
        # 独立运行模式（加载YOLOv5Seg模型和数据集）
        device = select_device(device, batch_size=batch_size)
        # 创建结果保存目录（YOLOv5专用，避免与VGG16结果混淆）
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        (save_dir / 'visualizations').mkdir(parents=True, exist_ok=True)  # 可视化结果子目录
        
        # 加载数据集配置（与VGG16版本一致，JSON格式）
        data_dict = check_dataset(data)
        nc = 1 if single_cls else int(data_dict['nc'])  # 类别数（CamVid为12）
        
        # 加载YOLOv5Seg模型（核心修改：替换为YOLOv5Seg，使用yolov5.yaml配置）
        model = YOLOv5Seg(cfg='yolov5.yaml', num_classes=nc).to(device)
        
        # 加载YOLOv5语义分割模型权重（核心修改：权重路径适配YOLOv5训练结果）
        if weights:
            ckpt = torch.load(weights, map_location=device)
            # 兼容训练保存的权重格式（可能包含'model'键或直接是模型权重）
            if 'model' in ckpt:
                ckpt = ckpt['model']
            model.load_state_dict(ckpt.float().state_dict())
        model.eval()  # 切换为评估模式
        
        # 检查图像尺寸（根据模型stride调整，确保下采样/上采样对齐）
        stride = model.stride  # 从YOLOv5Seg模型获取下采样倍数（[2,4,8,16,32]）
        imgsz = check_img_size(imgsz, s=stride)
        
        # 类别名称（与VGG16版本一致，CamVid-12类别）
        names = data_dict['names'] if not single_cls else ['class0']
        
        # 创建JSON格式数据加载器（与VGG16版本完全一致，确保数据输入统一）
        dataloader = create_json_segment_dataloader(
            img_dir=data_dict['val_img'],  # 验证集图像目录
            json_label_dir=data_dict['val_json'],  # 验证集JSON掩码目录（与VGG16共享）
            img_size=imgsz,
            batch_size=batch_size,
            augment=False,  # 评估时禁用数据增强
            workers=workers,
            shuffle=False  # 评估时不打乱数据
        )[0]

    # 初始化评估工具（混淆矩阵、计时）
    confusion_matrix = SegmentationConfusionMatrix(nc, ignore_index=11)  # 忽略unlabelled类别（索引11）
    seen = 0  # 已处理的样本数
    dt = Profile(), Profile(), Profile()  # 计时：预处理、推理、后处理

    # 开始验证（推理+评估）
    model.eval()
    pbar = tqdm(dataloader, desc=colorstr('val-yolov5-seg: '), bar_format=TQDM_BAR_FORMAT)  # 仅修改进度条标识
    for batch_i, (imgs, masks, paths) in enumerate(pbar):
        with dt[0]:  # 预处理计时
            # 数据类型与设备适配（确保与模型输入一致）
            if imgs.dtype == torch.uint8 or float(imgs.max()) > 1.0:
                imgs = imgs.to(device, non_blocking=True).float() / 255.0  # 归一化到[0,1]
            else:
                imgs = imgs.to(device, non_blocking=True).float()

            # 半精度推理（加速评估，与VGG16版本一致）
            if half:
                imgs = imgs.half()
            masks = masks.to(device, non_blocking=True).long()  # 掩码为类别索引（长整数类型）
            nb, _, height, width = imgs.shape  # nb: 批次大小

        with dt[1]:  # 推理计时
            # YOLOv5Seg模型推理（输出shape: (batch, nc, H, W)）
            preds = model(imgs, augment=augment) if not training else model(imgs)
            # 取通道维度最大值作为预测类别（语义分割核心后处理）
            pred_masks = torch.argmax(preds, dim=1)  # 输出shape: (batch, H, W)

        with dt[2]:  # 后处理计时
            # 逐样本处理（更新混淆矩阵+可视化）
            for si in range(nb):
                seen += 1
                path = Path(paths[si])  # 当前样本路径
                true_mask = masks[si].cpu().numpy()  # 真实掩码（numpy数组）
                pred_mask = pred_masks[si].cpu().numpy()  # 预测掩码（numpy数组）

                # 更新混淆矩阵（统计TP/FP/FN）
                confusion_matrix.process_batch(pred_mask, true_mask)

                # 可视化结果（每批次保存前3张，避免冗余）
                if plots and batch_i < 3 and si < 3:
                    visualize_path = save_dir / 'visualizations' / f'{path.stem}_val_yolov5.png'  # 仅修改文件名标识
                    visualize_results(imgs[si], pred_mask, true_mask, visualize_path, names)

        # 进度条更新（显示已处理样本数）
        pbar.set_postfix({'样本数': seen})

    # 计算最终评估指标（mIoU和各类别IoU）
    final_miou, per_cls_iou = confusion_matrix.compute_iou()

    # 打印评估结果（格式与VGG16版本完全一致，仅修改标题标识）
    LOGGER.info('\n' + '='*60)
    LOGGER.info(f"YOLOv5语义分割验证结果 | 验证集总数: {seen} 张图像")
    LOGGER.info(f"整体mIoU: {final_miou:.4f}")
    LOGGER.info('-'*60)
    LOGGER.info(f"{'类别名称':<15} {'IoU值':<10}")
    LOGGER.info('-'*25)
    cls_idx = 0  # 跳过忽略类别的IoU索引
    for cls in range(nc):
        if cls != 11:  # 不显示unlabelled类别（索引11）
            LOGGER.info(f"{names[cls]:<15} {per_cls_iou[cls_idx]:.4f}")
            cls_idx += 1
    LOGGER.info('='*60)

    # 保存混淆矩阵图（独立运行模式下）
    if plots and not training:
        confusion_matrix.plot(save_dir, names)

    # 打印推理速度（预处理/推理/后处理耗时）
    t = tuple(x.t / seen * 1000 for x in dt)  # 转换为每张图像的毫秒数
    LOGGER.info(f"推理速度: 预处理 {t[0]:.1f}ms/张, 推理 {t[1]:.1f}ms/张, 后处理 {t[2]:.1f}ms/张")

    # 打印结果保存路径（独立运行模式下）
    if not training:
        LOGGER.info(f"验证结果保存至: {colorstr('bold', save_dir)}")

    # 返回评估指标（mIoU, 各类别IoU, 速度），与VGG16版本输出格式一致
    return (final_miou,), per_cls_iou, t


# -------------------------- 命令行参数解析（仅修改默认路径，适配YOLOv5） --------------------------
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'CamVid/data.yaml', 
                      help='数据集配置文件（与VGG16版本共享，JSON格式）')
    parser.add_argument('--weights', type=str, 
                      default='runs/train-yolov5-seg/weights/best.pt',  # 核心修改：YOLOv5最佳权重路径
                      help='YOLOv5语义分割模型权重路径')
    parser.add_argument('--batch-size', type=int, default=8, help='批量大小（与VGG16版本一致）')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, 
                      help='输入图像尺寸（与VGG16版本一致）')
    parser.add_argument('--device', default='', help='设备配置，例如 0 或 cpu（与VGG16版本一致）')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数（与VGG16版本一致）')
    parser.add_argument('--single-cls', action='store_true', help='单类别数据集（CamVid为多类别，默认关闭）')
    parser.add_argument('--augment', action='store_true', help='增强推理（评估时默认关闭）')
    parser.add_argument('--verbose', action='store_true', help='详细输出每个类别的IoU（与VGG16版本一致）')
    parser.add_argument('--project', default=ROOT / 'runs/val-yolov5-seg', 
                      help='YOLOv5验证结果保存项目路径（核心修改：区分VGG16）')
    parser.add_argument('--name', default='exp', help='实验名称（默认exp）')
    parser.add_argument('--exist-ok', action='store_true', help='允许覆盖现有实验目录（与VGG16版本一致）')
    parser.add_argument('--half', action='store_true', help='使用FP16半精度推理（加速评估）')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # 验证数据集配置文件有效性
    print_args(vars(opt))  # 打印解析后的参数
    return opt


# -------------------------- 主函数（与VGG16版本完全一致） --------------------------
def main(opt):
    check_requirements()  # 检查依赖包是否安装
    run(** vars(opt))  # 调用核心验证函数


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)