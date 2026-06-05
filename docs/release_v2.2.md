# V2.2 版本说明：YOLO Segmentation 评价增强

## 1. 版本定位

V2.2 在 V2.1 图像级评价基础上，新增像素级 IoU/Dice 分割质量评价和 confidence 阈值实验，补全 YOLO segmentation 的专业评价体系。

## 2. 新增评价

- IoU / Dice 像素级分割质量（mean IoU_tp=0.553, mean Dice_tp=0.698）
- 5 个 confidence 阈值的 Precision-Recall 实验
- 4 联可视化案例（原图 | GT mask | YOLO pred | overlay）

## 3. 核心结果

### 图像级（conf=0.25）

| TP | TN | FP | FN | Prec | Rec | F1 |
|----|----|----|----|------|-----|-----|
| 66 | 894 | 0 | 44 | 100% | 60% | 0.750 |

### 最佳 F1（conf=0.05）

| TP | TN | FP | FN | Prec | Rec | **F1** |
|----|----|----|----|------|-----|---------|
| 85 | 885 | 9 | 25 | 90.4% | 77.3% | **0.833** |

### 分割质量

| mean IoU (TP) | mean Dice (TP) |
|---------------|----------------|
| 0.553 | 0.698 |

## 4. 与 V2.1 的关系

V2.1 完成了图像级 OK/NG 评价。V2.2 在此基础上增加了像素级分割评价和置信度实验，让评估更完整。

## 5. 当前局限

- 仅 3 epochs，训练不充分
- 44 张 FN 仍存在
- 分割 IoU 有提升空间

## 6. V2.3 计划

- epochs=30 正式训练、模型对比、最终 OpenCV vs YOLO 报告
