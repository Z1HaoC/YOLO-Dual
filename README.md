# 语义分割模型框架

一个支持多种Backbone和Head结构的语义分割模型训练与验证框架，集成了不同骨干网络和分割头的实现，适用于科研实验与工程落地。


## 📌 项目简介
本仓库提供了基于多种Backbone（骨干网络）和Head（分割头）的语义分割模型训练与验证代码，支持灵活配置网络结构，方便进行不同模型架构的性能对比与改进实验。

核心特点：
- 支持多种经典Backbone（ResNet、MobileNet、EfficientNet等）
- 提供多种分割头设计（基础分割头、YOLOv5风格分割头、带注意力机制的分割头等）
- 包含完整的训练/验证流程，支持损失函数定制（如Dice Loss）
- 适配不同输入尺寸，自动处理特征对齐与上采样


## 🚀 支持的模型结构

### 🔹 Backbone（骨干网络）
| 模型名称 | 说明 | 参考配置 |
|---------|------|---------|
| ResNet18/34/50 | 经典残差网络，适合中等算力场景 | `unet-lite/Resnet50/` |
| MobileNetV2/V3 | 轻量级网络，适合移动端部署 | 基于`torchvision.models`实现 |
| EfficientNet B0/B1/V2_s | 高效网络，兼顾精度与速度 | 对齐`torchvision.models`预训练权重 |
| ConvNeXt Tiny | 现代卷积网络，性能优异 | `models/backbone/convnext_tiny.yaml` |
| YOLOv5 Backbone | 包含C3、C3-DCN、SPPF等模块 | `unet-lite/yolo7-seg/` |
| YOLOv8 Backbone | 包含C3、C3-DCN、SPPF等模块 | `unet-lite/yolo8-seg/` |
| YOLOv9 Backbone | 包含C3、C3-DCN、SPPF等模块 | `unet-lite/yolo9-seg/` |  

> 所有Backbone支持加载ImageNet预训练权重，加速模型收敛。


### 🔸 Head（分割头）
| 分割头类型 | 特点 | 代码位置 |
|-----------|------|---------|
| 基础多尺度分割头 | 融合多尺度特征，通过卷积与上采样输出像素分类 | `segment/train.py`（`SegmentHead`） |
| YOLOv5风格分割头 | 结合Upsample/Concat操作，支持跳跃连接 | `unet-lite/yolo5-seg/seg_diceloss_yolov5.py` |
| ResNet系列分割头 | 适配ResNet backbone的特征融合结构 | `unet-lite/Resnet50/seg_diceloss_Resnet50.py` |

后续还会增加Backbone 和 Head

> 支持自定义损失函数Diceloss/Jaccardloss(IoU Loss)/Entrophy Loss，适合样本不平衡场景。


## 🛠️ 环境配置
```bash
# 安装依赖
pip install torch torchvision opencv-python numpy thop PyYAML
```


## 🔍 使用方法

unet-lite中的方式已全部得到验证，优先使用
tensorboard已配置可以直接可视化训练图片的实时分割训练结果

### 训练模型
```bash
# 示例：使用ResNet50 backbone + Dice Loss训练
python unet-lite/Resnet50/seg_diceloss_Resnet50.py \
  --data your_dataset.yaml \
  --epochs 100 \
  --batch-size 16 \
  --img-size 640 640 \
  --weights pretrained  # 加载预训练权重
```

```bash
# 示例：使用YOLOv5 backbone训练
python unet-lite/yolo5-seg/seg_diceloss_yolov5.py \
  --cfg models/yolov5_seg.yaml \
  --data your_dataset.yaml \
  --epochs 50 \
  --batch-size 8
```


### 验证模型
```bash
# 示例：验证训练好的模型
python segment/val.py \
  --weights runs/train/exp/weights/best.pt \
  --data your_dataset.yaml \
  --img-size 640 640
```


## 📊 模型性能参考
| Backbone | 输入尺寸 | 参数量(M) | GFLOPs | 备注 |
|---------|---------|----------|--------|------|
| ResNet50 | 640x640 | ~25 | ~45 | 适合高精度场景 |
| MobileNetV3 Small | 640x640 | ~4.7 | ~10 | 轻量级部署首选 |
| YOLOv5 Backbone | 640x640 | ~13 | ~20 | 兼顾速度与精度 |


## 📚 参考资源
- 骨干网络参考：[torchvision.models](https://pytorch.org/vision/stable/models.html)
- 语义分割技巧：[YOLOv5数据增强](https://blog.csdn.net/OpenDataLab/article/details/127788561)
- 损失函数：[Dice Loss实现](https://github.com/WangRongsheng/BestYOLO)


## 🤝 贡献
欢迎提交PR扩展新的Backbone或分割头结构，一起完善该框架！

[![Star History](https://api.star-history.com/svg?repos=your_username/your_repo&type=Date)](https://star-history.com/#your_username/your_repo&Date)
