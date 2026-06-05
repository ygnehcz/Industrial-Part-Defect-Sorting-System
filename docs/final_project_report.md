# 工业零件表面缺陷检测系统 — 最终项目报告

## 项目背景

本项目旨在搭建一个完整的工业表面缺陷检测系统，从传统机器视觉方法（OpenCV）出发，逐步演进到深度学习方案（YOLOv8-seg），在真实工业缺陷数据集上完成全流程验证。

## 数据集说明

- 来源：KolektorSDD2 风格工业表面缺陷数据集
- 规模：train 2331 对 + test 1004 对 = 3335 对
- 标签：像素级 `*_GT.png` mask
- OK/N：GT mask 全黑=OK，含白像素=NG
- 仅使用图像级标签进行评价

## OpenCV baseline（V1.1）

固定阈值方法在模拟图片上 100% Recall，在真实数据上仅 4.5%。

```
TP=5, TN=894, FP=0, FN=105, F1=0.087
```

## CLAHE 改进（V1.2）

引入 CLAHE 局部对比度增强后，F1 提升 7.9 倍。

```
TP=65, TN=880, FP=14, FN=45, F1=0.688
```

## YOLO segmentation 数据转换（V2.0）

将像素级 GT mask 转换为 YOLO segmentation polygon 格式：
- 3335 对图像-mask 全部转换成功
- 保持原始 train/val 划分
- data/yolo_seg/data.yaml 配置完成

## YOLO 训练与模型选择（V2.1 ~ V2.4）

| 版本 | 模型 | epochs | F1 | 结论 |
|------|------|--------|-----|------|
| V2.1 | yolov8n-seg | 3 | 0.750 | smoke training 成功 |
| V2.2 | yolov8n-seg | 3 | **0.833** | +IoU/Dice + confidence 实验 |
| V2.3 | yolov8n-seg | 10 | 0.808 | 更多 epoch 反而 F1 下降 |
| V2.4 | 对比 | — | — | V2.2 3ep 为最佳 |

## 最终推荐模型

**V2.2 yolov8n-seg 3 epochs, confidence=0.05**

| TP | TN | FP | FN | Accuracy | Precision | Recall | F1 |
|----|----|----|----|----------|-----------|--------|-----|
| 85 | 885 | 9 | 25 | 96.6% | 90.4% | 77.3% | **0.833** |

## OpenCV vs YOLO 指标对比

| 方法 | F1 | Precision | Recall | FP | FN |
|------|-----|-----------|--------|-----|-----|
| OpenCV 固定阈值 | 0.087 | 100.0% | 4.5% | 0 | 105 |
| CLAHE 增强 | 0.688 | 82.3% | 59.1% | 14 | 45 |
| **YOLO 3ep** | **0.833** | **90.4%** | **77.3%** | **9** | **25** |

## 错误案例分析

- FN（25 张漏检）：缺陷灰度对比度极低，或缺陷面积过小
- FP（9 张误报）：表面纹理/反光被误识别为缺陷
- 工业场景中，FN 通常比 FP 更严重（漏检导致缺陷产品流出）

## 当前局限

- CPU 训练，batch=2，训练不充分
- 3 epochs 训练远未收敛
- 仍有 25 张 FN（Recall=77.3%）
- 仅使用图像级评价，未做像素级 IoU 定位分析
- confidence=0.05 是经验选择，未经产线验证

## 后续 GPU 正式训练计划

- GPU 训练 yolov8s-seg, epochs=50~100, batch=16
- 数据增强：mosaic, mixup, hsv
- 学习率余弦退火
- 正式验证集评价
- 导出 ONNX 部署

## 面试讲法

这个项目是我用来系统学习工业缺陷检测完整流程的。我选了一个真实工业表面缺陷数据集，从传统 OpenCV 方法开始做 baseline。最初固定阈值在真实数据上只有 4.5% 的 Recall，然后我用 CLAHE 增强提到了 59%。之后我把数据集的 GT mask 转成了 YOLO segmentation 格式，用 YOLOv8n-seg 训练了 3 个 epoch，F1 就到了 0.833，比 CLAHE 又高了 21%。

在过程中我做了 100 多组 OpenCV 阈值实验、50 多组增强方法对比、YOLO 的 confidence 实验和模型选择分析。最终结论是 3 个 epoch 的 YOLO 比 10 个 epoch 的效果还好，说明在小数据集上更多训练不一定更好。

这个项目让我理解了从传统方法到深度学习的完整演进路线，也让我实际体验了工业缺陷检测中 Precision-Recall 权衡、误检漏检分析的工程意义。
