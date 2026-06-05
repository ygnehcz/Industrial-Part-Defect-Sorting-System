# YOLO 训练对比分析

## 1. 训练设置

| 参数 | V2.1 (3ep) | V2.3 (10ep) |
|------|-----------|-------------|
| 模型 | yolov8n-seg | yolov8n-seg |
| Epochs | 3 | 10 |
| imgsz | 320 | 320 |
| batch | 2 | 2 |
| device | CPU | CPU |
| 耗时 | ~37min | ~29min |

## 2. 训练指标对比

| 指标 | V2.1 (ep3) | V2.3 (ep10) |
|------|-----------|-------------|
| mAP50(M) | 0.335 | 0.294 |
| mAP50-95(M) | 0.136 | 0.143 |
| Mask Precision | 0.579 | 0.323 |
| Mask Recall | 0.269 | 0.370 |

## 3. 图像级评价对比（conf=0.05）

| 版本 | TP | TN | FP | FN | Prec | Rec | **F1** |
|------|----|----|----|----|------|-----|---------|
| V1.2 CLAHE | 65 | 880 | 14 | 45 | 82.3% | 59.1% | 0.688 |
| V2.2 (3ep) | 85 | 885 | 9 | 25 | 90.4% | 77.3% | **0.833** |
| V2.3 (10ep) | 86 | 877 | 17 | 24 | 83.5% | 78.2% | 0.808 |

## 4. 分割质量对比

| 指标 | V2.2 (3ep) | V2.3 (10ep) |
|------|-----------|-------------|
| mean IoU (TP) | 0.553 | 0.474 |
| mean Dice (TP) | 0.698 | 0.624 |

## 5. F1 对比图

![F1](docs/images/yolo_training_charts/yolo_f1_comparison.png)

![PR](docs/images/yolo_training_charts/yolo_precision_recall_comparison.png)

![IoU](docs/images/yolo_training_charts/yolo_iou_dice_comparison.png)

## 6. 分析

- 10 epochs 的 Recall（78.2%）略高于 3 epochs（77.3%）
- 但 Precision 下降（90.4%→83.5%），FP 从 9 增至 17
- F1 从 0.833 降至 0.808——更多 epoch 带来了更多 FP
- 分割质量也轻微下降：IoU 0.553→0.474, Dice 0.698→0.624
- **可能原因**：模型开始过拟合训练集，或学习率衰减导致的不稳定
- **所有 YOLO 版本均大幅优于 OpenCV CLAHE（F1 0.688）**

## 7. 局限

- CPU 训练，batch=2，训练不够充分
- 10 epochs 可能仍不够，但出现了精度下降迹象
- 需要 GPU + 更大 batch 做正式训练
- YOLO segmentation 仍有提升空间

## 8. 最终结论

经过 V1.1 到 V2.3 的迭代，项目在真实工业缺陷数据上的检测能力实现了显著提升：

| 方法 | F1 | 提升 |
|------|-----|------|
| OpenCV 固定阈值 (V1.1) | 0.087 | — |
| CLAHE 增强 (V1.2) | 0.688 | +691% |
| YOLO 3ep (V2.2) | 0.833 | +21% vs CLAHE |
| YOLO 10ep (V2.3) | 0.808 | +17% vs CLAHE |

**YOLO 3 epoch 即超越 CLAHE，在 Recall 和 Precision 之间取得最佳平衡。**
