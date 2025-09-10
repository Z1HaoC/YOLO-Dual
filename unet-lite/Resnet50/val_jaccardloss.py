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

from models.common import DetectMultiBackend
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size,
                           check_requirements, check_yaml, colorstr, increment_path, print_args)
from utils.torch_utils import select_device, smart_inference_mode
from utils.segment.dataloaders import create_dataloader


# CamVid颜色映射表(RGB)
CAMVID_COLORS = [
    [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
    [60, 40, 222], [128, 128, 0], [192, 128, 128], [64, 64, 128],
    [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 0]  # 最后一个是unlabelled
]


class SegmentationConfusionMatrix:
    """语义分割专用混淆矩阵，处理一维类别索引"""
    def __init__(self, num_classes, ignore_index=11):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def process_batch(self, preds, targets):
        """
        处理单个批次
        preds: 预测掩码 (H, W) 或展平后的一维数组，值为类别索引
        targets: 真实掩码 (H, W) 或展平后的一维数组，值为类别索引
        """
        # 展平为一维数组
        if preds.ndim > 1:
            preds = preds.flatten()
        if targets.ndim > 1:
            targets = targets.flatten()

        # 过滤忽略的类别
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
                
            # 计算交并比
            true_positive = self.matrix[cls, cls]
            false_positive = self.matrix[:, cls].sum() - true_positive
            false_negative = self.matrix[cls, :].sum() - true_positive
            union = true_positive + false_positive + false_negative
            
            if union == 0:
                ious.append(0.0)  # 没有该类别的样本
            else:
                ious.append(true_positive / union)
        
        return np.mean(ious), ious

    def plot(self, save_dir, names):
        """可视化混淆矩阵"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(self.matrix, cmap='Blues')
        
        # 标注数值
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
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png')
        plt.close()


def visualize_results(img, pred_mask, true_mask, save_path, names):
    """可视化原图、真实掩码和预测掩码"""
    # 转换图像格式 (tensor -> numpy)
    img_np = img.permute(1, 2, 0).cpu().numpy() * 255  # 反归一化
    img_np = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # 为掩码上色
    pred_colored = np.zeros_like(img_np)
    true_colored = np.zeros_like(img_np)
    
    for cls in range(len(CAMVID_COLORS)):
        pred_colored[pred_mask == cls] = CAMVID_COLORS[cls]
        true_colored[true_mask == cls] = CAMVID_COLORS[cls]
    
    # 叠加掩码到原图
    pred_overlay = cv2.addWeighted(img_np, 0.5, pred_colored, 0.5, 0)
    true_overlay = cv2.addWeighted(img_np, 0.5, true_colored, 0.5, 0)
    
    # 创建标题
    h, w = img_np.shape[:2]
    title = np.ones((50, w*3, 3), dtype=np.uint8) * 255
    cv2.putText(title, '原图', (w//3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(title, '真实掩码', (w + w//3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(title, '预测掩码', (2*w + w//3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    
    # 拼接图像
    combined = np.vstack((title, np.hstack((img_np, true_overlay, pred_overlay))))
    cv2.imwrite(str(save_path), combined)


@smart_inference_mode()
def run(
        data,
        weights=None,
        batch_size=8,
        imgsz=360,  # CamVid图像高度
        device='',
        workers=4,
        single_cls=False,
        augment=False,
        verbose=False,
        project=ROOT / 'runs/val-seg',
        name='exp',
        exist_ok=False,
        half=True,
        dnn=False,
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        compute_loss=None,
):
    # 初始化模式（训练中调用或独立运行）
    training = model is not None
    if training:
        # 训练过程中调用（复用模型和数据加载器）
        device = next(model.parameters()).device
        nc = model.num_classes
        names = model.names
        
        # 确保模型与输入数据类型匹配
        if half:
            model = model.half()
        else:
            model = model.float()
    else:
        # 独立运行模式
        device = select_device(device, batch_size=batch_size)
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        (save_dir / 'visualizations').mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride = model.stride
        imgsz = check_img_size(imgsz, s=stride)  # 检查图像尺寸
        
        # 加载数据集配置
        data = check_dataset(data)
        nc = 1 if single_cls else int(data['nc'])
        names = data['names'] if not single_cls else ['class0']
        
        # 创建数据加载器
        dataloader = create_dataloader(
            data['val'],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=0.5,
            rect=True,
            workers=workers,
            prefix=colorstr('val: ')
        )[0]

    # 初始化评估工具
    confusion_matrix = SegmentationConfusionMatrix(nc, ignore_index=11)  # 忽略unlabelled类别
    seen = 0
    total_miou = 0.0
    class_ious = [0.0] * nc
    class_counts = [0] * nc
    dt = Profile(), Profile(), Profile()  # 计时：预处理、推理、后处理

    # 开始验证
    model.eval()
    pbar = tqdm(dataloader, desc=colorstr('val: '), bar_format=TQDM_BAR_FORMAT)
    for batch_i, batch_data in enumerate(pbar):
        # 解析批次数据，兼容3元素和5元素格式
        if len(batch_data) == 3:
            im, masks, paths = batch_data
            targets = torch.zeros((0, 5), device=device)
            shapes = [(im.shape[2:], (0,0,0,0)) for _ in range(im.shape[0])]
        elif len(batch_data) == 5:
            im, targets, paths, shapes, masks = batch_data
        else:
            raise ValueError(f"数据加载器返回{len(batch_data)}个元素，需要3或5个")

        with dt[0]:  # 预处理计时
            # 数据预处理
            if im.dtype == torch.uint8 or float(im.max()) > 1.0:
                # 是 0-255 范围的数据
                im = im.to(device, non_blocking=True).float() / 255.0
            else:
                # 已经在 [0,1]（float）
                im = im.to(device, non_blocking=True).float()

            if half:
                im = im.half()
            masks = masks.to(device, non_blocking=True).long()  # 掩码转为长整数类型
            nb, _, height, width = im.shape

        with dt[1]:  # 推理计时
            # 模型推理（语义分割输出：(batch, classes, H, W)）
            preds = model(im, augment=augment) if not training else model(im)
            # 取概率最大的类别作为预测结果
            pred_masks = torch.argmax(preds, dim=1)  # 形状：(batch, H, W)

        with dt[2]:  # 后处理计时
            # 处理每个样本
            for si in range(nb):
                seen += 1
                path = Path(paths[si])
                true_mask = masks[si].cpu().numpy()  # 真实掩码
                pred_mask = pred_masks[si].cpu().numpy()  # 预测掩码

                # 更新混淆矩阵
                confusion_matrix.process_batch(pred_mask, true_mask)

                # 计算当前样本的mIoU
                miou, ious = confusion_matrix.compute_iou()
                total_miou += miou

                # 累加每个类别的IoU
                cls_idx = 0
                for cls in range(nc):
                    if cls != 11:  # 跳过unlabelled类别
                        class_ious[cls] += ious[cls_idx]
                        class_counts[cls] += 1
                        cls_idx += 1

                # 可视化（每批次保存前3张）
                if plots and batch_i < 3 and si < 3:
                    visualize_path = save_dir / 'visualizations' / f'{path.stem}_val.png'
                    visualize_results(im[si], pred_mask, true_mask, visualize_path, names)

        # 显示进度
        pbar.set_postfix(
            {'样本数': seen, '平均mIoU': f'{total_miou/seen:.4f}'}
        )

    # 计算最终指标
    final_miou, per_cls_iou = confusion_matrix.compute_iou()
    avg_class_ious = []
    cls_idx = 0
    for cls in range(nc):
        if cls != 11 and class_counts[cls] > 0:
            avg_class_ious.append(class_ious[cls] / class_counts[cls])
        else:
            avg_class_ious.append(0.0)

    # 打印结果
    LOGGER.info('\n' + '='*50)
    LOGGER.info(f"验证集总数: {seen} 张图像")
    LOGGER.info(f"整体mIoU: {final_miou:.4f}")
    LOGGER.info('-'*50)
    LOGGER.info(f"{'类别':<15} {'IoU':<10}")
    LOGGER.info('-'*25)
    cls_idx = 0
    for cls in range(nc):
        if cls != 11:  # 不显示unlabelled类别
            LOGGER.info(f"{names[cls]:<15} {avg_class_ious[cls_idx]:.4f}")
            cls_idx += 1
    LOGGER.info('='*50)

    # 保存混淆矩阵
    if plots and not training:
        confusion_matrix.plot(save_dir, names)

    # 打印速度信息
    t = tuple(x.t / seen * 1000 for x in dt)
    LOGGER.info(f"速度: 预处理 {t[0]:.1f}ms, 推理 {t[1]:.1f}ms, 后处理 {t[2]:.1f}ms 每张图像")

    # 保存结果
    if not training:
        LOGGER.info(f"结果保存至: {colorstr('bold', save_dir)}")

    return (final_miou,), avg_class_ious, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / '/mnt/afs/senserobot22_chenzihao2/BestYOLO/CamVid/data.yaml', help='数据集配置文件')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='模型路径')
    parser.add_argument('--batch-size', type=int, default=8, help='批量大小')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=360, help='输入图像尺寸')
    parser.add_argument('--device', default='', help='设备，例如 0 或 cpu')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--single-cls', action='store_true', help='单类别数据集')
    parser.add_argument('--augment', action='store_true', help='增强推理')
    parser.add_argument('--verbose', action='store_true', help='详细输出每个类别的IoU')
    parser.add_argument('--project', default=ROOT / 'runs/val-seg', help='保存项目路径')
    parser.add_argument('--name', default='exp', help='实验名称')
    parser.add_argument('--exist-ok', action='store_true', help='允许覆盖现有实验')
    parser.add_argument('--half', action='store_true', help='使用FP16半精度推理')
    parser.add_argument('--dnn', action='store_true', help='使用OpenCV DNN进行ONNX推理')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements()
    run(** vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
