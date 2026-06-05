# V2.1 版本说明：YOLO segmentation 初步训练与验证

## 1. 版本定位

V2.1 在 V2.0 数据集准备的基础上，完成 YOLOv8-seg 的 smoke training（3 epochs），并进行图像级 OK/NG 评价，与 V1.2 CLAHE baseline 对比。

## 2. 训练是否完成

✅ Smoke training 完成（3 epochs，CPU，约 37 分钟）

## 3. 使用模型

- 模型：yolov8n-seg.pt（从 COCO 预训练权重迁移学习）
- 输出：outputs/yolo_seg/runs/yolov8n_seg_smoke/weights/best.pt

## 4. 初步评价结果

| 方法 | TP | TN | FP | FN | Precision | Recall | F1 |
|------|----|----|----|----|-----------|--------|-----|
| CLAHE (V1.2) | 65 | 880 | 14 | 45 | 82.3% | 59.1% | 0.688 |
| **YOLO seg (3ep)** | **66** | **894** | **0** | **44** | **100.0%** | **60.0%** | **0.750** |

## 5. 与 OpenCV baseline 对比

YOLO 仅 3 个 epoch 就在所有指标上超越 CLAHE：
- F1：0.688 → 0.750
- Precision：82.3% → 100.0%
- Recall：59.1% → 60.0%
- FP：14 → 0

## 6. 后续 V2.2 正式训练计划

- epochs=50~100，GPU 训练
- 重新评估图像级与像素级指标
- 绘制 Precision-Recall 对比曲线
- 分析不同 epoch 下的 Precision-Recall 权衡
