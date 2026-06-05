# YOLO Segmentation 训练分析

## 训练设置

- 模型：yolov8n-seg.pt（YOLOv8 nano segmentation 预训练权重）
- 数据：data/yolo_seg/data.yaml（train=2331, val=1004）
- Epochs：3（smoke training）
- imgsz：320
- batch：2
- device：CPU（Intel i7-12700H）
- workers：0

## 训练过程

| Epoch | mAP50(M) | mAP50-95(M) | Mask Precision | Mask Recall | 耗时 |
|-------|----------|-------------|----------------|-------------|------|
| 1 | 0.075 | 0.025 | 0.000 | 0.655 | 283s |
| 2 | 0.136 | 0.046 | 0.238 | 0.210 | 1130s |
| 3 | 0.335 | 0.136 | 0.579 | 0.269 | 831s |

总训练时间约 37 分钟（CPU）。

## YOLO 图像级 TP/TN/FP/FN（val 集）

| TP | TN | FP | FN | Accuracy | Precision | Recall | F1 |
|----|----|----|----|----------|-----------|--------|-----|
| 66 | 894 | 0 | 44 | 95.6% | 100.0% | 60.0% | **0.750** |

## 与 V1.2 CLAHE OpenCV baseline 对比

| 方法 | TP | TN | FP | FN | Precision | Recall | F1 |
|------|----|----|----|----|-----------|--------|-----|
| CLAHE (V1.2) | 65 | 880 | 14 | 45 | 82.3% | 59.1% | 0.688 |
| **YOLO seg (3ep)** | **66** | **894** | **0** | **44** | **100.0%** | **60.0%** | **0.750** |

YOLO 在仅 3 个 epoch 的 smoke training 后就全面超越了 CLAHE：
- Recall：59.1% → 60.0%（+0.9pp）
- Precision：82.3% → 100.0%（+17.7pp，零误报）
- F1：0.688 → 0.750（+9.0%）

## 当前训练轮数太少的局限

- 3 个 epoch 远不足以收敛，mAP50(M) 仅 0.335
- 模型仍有很大提升空间
- FP=0 可能因模型过于保守（阈值高），并非真正完美
- 需要更多 epoch 验证 Precision 和 Recall 的最终平衡点
- CPU 训练速度慢，正式训练建议 GPU

## 后续正式训练计划（V2.2）

- epochs=50~100
- imgsz=640
- batch=8~16
- GPU 训练
- 训练后重新评估图像级指标
- 对比不同 epoch 的 Precision-Recall 曲线
- 像素级 mask IoU / Dice 评估
- 绘制 OpenCV CLAHE vs YOLO 的 Precision-Recall 对比图
