# YOLO 模型选择分析

## 1. 为什么 10 epochs 不一定优于 3 epochs

V2.2（3ep）和 V2.3（10ep）使用相同数据、相同 yolov8n-seg 模型，仅训练轮数不同。结果 3ep 反而更优：

- 3ep 最佳 F1=0.833（conf=0.05）
- 10ep 最佳 F1=0.826（conf=0.10）

原因分析：
- CPU 训练 batch=2 可能不够稳定，更多 epoch 导致过拟合
- 学习率衰减后模型在训练集上过拟合，泛化能力下降
- 10ep 的 FP 更多（conf=0.05 时 FP=17 vs 9），说明模型过度敏感

## 2. V2.2 vs V2.3 在不同 confidence 下的指标

![F1 Curve](docs/images/yolo_model_selection_charts/model_conf_f1_curve.png)

| conf | V2.2 F1 | V2.2 Rec | V2.3 F1 | V2.3 Rec |
|------|---------|----------|---------|----------|
| 0.01 | 0.712 | 86.4% | 0.633 | 85.5% |
| 0.03 | 0.832 | 80.9% | 0.753 | 81.8% |
| 0.05 | **0.833** | 77.3% | 0.808 | 78.2% |
| 0.10 | 0.817 | 70.9% | 0.826 | 75.5% |
| 0.25 | 0.750 | 60.0% | 0.813 | 69.1% |
| 0.50 | 0.616 | 44.5% | 0.736 | 58.2% |

![PR Curve](docs/images/yolo_model_selection_charts/model_conf_precision_recall_curve.png)

## 3. 最佳 F1 模型

**V2.2 3ep, conf=0.05**

| TP | TN | FP | FN | Prec | Rec | **F1** |
|----|----|----|----|------|-----|---------|
| 85 | 885 | 9 | 25 | 90.4% | 77.3% | **0.833** |

![Confusion](docs/images/yolo_model_selection_charts/best_model_confusion_matrix.png)

## 4. Recall >= 80% 时最佳

**V2.2 3ep, conf=0.03**

| TP | TN | FP | FN | Prec | Rec | F1 |
|----|----|----|----|------|-----|-----|
| 89 | 879 | 15 | 21 | 85.6% | 80.9% | 0.832 |

如果需要 Recall >= 80%，可使用 V2.2 conf=0.03。

## 5. FP/FN 错误案例分析

![both_TP](docs/images/yolo_model_comparison_cases/both_TP_01.png)
两个模型都正确检出的大缺陷。

![both_FN](docs/images/yolo_model_comparison_cases/both_FN_01.png)
两个模型都漏检的困难样本。

![v22_better](docs/images/yolo_model_comparison_cases/v22_better_01.png)
V2.2 检出但 V2.3 漏检——10ep 过度保守。

![v23_better](docs/images/yolo_model_comparison_cases/v23_better_01.png)
V2.3 检出但 V2.2 漏检——10ep 个别样本有提升。

## 6. 当前推荐模型

**V2.2 yolov8n-seg 3ep, conf=0.05**

- F1 最高（0.833）
- FP 控制好（仅 9 个）
- 推理速度快（nano 模型）
- 与 CLAHE baseline 相比：F1 0.688→0.833

## 7. 是否建议继续训练 20/30 epochs

**不建议在本机 CPU 继续训练。** 理由：
- 3ep→10ep 已经出现 F1 下降
- 更深训练需要 GPU + 更大 batch + 学习率调度
- 当前 3ep F1=0.833 已经是一个很好的 baseline

## 8. 后续建议

- GPU 训练：epochs=50, batch=16, 使用 cosine lr scheduler
- 数据增强：mosaic, mixup 等
- 更大模型：yolov8s-seg 或 yolov8m-seg
- 正式验证集评估与 OpenCV 对比报告
