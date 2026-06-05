# V2.3 版本说明：YOLO 10-epoch 正式训练

## 1. 版本定位

V2.3 完成 YOLOv8n-seg 的 10-epoch 正式训练，与 3-epoch smoke training 进行系统对比。

## 2. 训练结果

| 指标 | 3-epoch | 10-epoch |
|------|---------|----------|
| mAP50(M) | 0.335 | 0.294 |
| mAP50-95(M) | 0.136 | 0.143 |
| 图像级 F1 (conf=0.05) | 0.833 | 0.808 |
| 图像级 Recall | 77.3% | 78.2% |
| mean IoU (TP) | 0.553 | 0.474 |

## 3. 与 V2.2 对比

10-epoch 训练在 Recall 上略有提升（77.3%→78.2%），但 Precision 和 IoU 轻微下降，F1 从 0.833 降至 0.808。可能因过拟合或学习率策略导致。

## 4. 与 OpenCV CLAHE 对比

所有 YOLO 版本（F1 0.75~0.833）均大幅优于 OpenCV CLAHE（F1 0.688）。

## 5. 局限

- CPU 训练限制，batch=2
- 分割质量有波动
- 仍有 24 张 FN

## 6. 后续

- GPU 训练（epochs=50+, batch=16）
- 模型对比（nano vs small vs medium）
- 部署方案探索
