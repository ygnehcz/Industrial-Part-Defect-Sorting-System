# V2.2 YOLO Segmentation 评价增强分析

## 1. V2.1 图像级评价回顾

| TP | TN | FP | FN | Accuracy | Precision | Recall | F1 |
|----|----|----|----|----------|-----------|--------|-----|
| 66 | 894 | 0 | 44 | 95.6% | 100.0% | 60.0% | 0.750 |

## 2. 为什么需要 IoU / Dice

图像级 OK/NG 只能判断"有没有检出缺陷"，不能评价"检出的区域有多准"。因为数据集提供像素级 GT mask，应补充分割质量评价：

- **IoU**（交并比）：预测 mask 与 GT mask 的重合面积 / 并集面积
- **Dice**（F1 的像素版）：2 × 重合面积 / (预测面积 + GT 面积)

## 3. IoU / Dice 评价结果

| 指标 | 数值 |
|------|------|
| mean IoU (NG 图) | 0.332 |
| mean Dice (NG 图) | 0.419 |
| **mean IoU (TP 图)** | **0.553** |
| **mean Dice (TP 图)** | **0.698** |
| TP mean pred_area | 4,172 |
| TP mean gt_area | 3,876 |
| FN mean gt_area | 5,242 |

**分析**：
- TP 图（成功检出缺陷）的 mean Dice=0.698，说明预训练 mask 与真实缺陷区域有约 70% 的重合
- mean IoU=0.553，分割边界精度仍有提升空间
- FN 图的 GT 面积（5,242）反而大于 TP（3,876），再次说明漏检与面积无关
- 仅 3 个 epoch 的训练，分割质量可期

## 4. confidence 阈值实验

| conf | TP | TN | FP | FN | Precision | Recall | F1 |
|------|----|----|----|----|-----------|--------|-----|
| 0.05 | 85 | 885 | 9 | 25 | 90.4% | 77.3% | **0.833** |
| 0.10 | 78 | 891 | 3 | 32 | 96.3% | 70.9% | 0.817 |
| 0.25 | 66 | 894 | 0 | 44 | 100.0% | 60.0% | 0.750 |
| 0.50 | 49 | 894 | 0 | 61 | 100.0% | 44.5% | 0.616 |
| 0.75 | 39 | 894 | 0 | 71 | 100.0% | 35.5% | 0.524 |

**关键发现**：
- 默认 conf=0.25 过于保守（FP=0 但漏检多）
- conf=0.05 时 F1 达到 **0.833**，Recall 提升到 77.3%
- 代价是引入 9 个 FP（Precision 从 100% 降至 90.4%）
- 工业场景中 conf 可根据容忍度调整：宁可漏检则高 conf，宁可误报则低 conf

## 5. 典型案例分析

### TP 案例（66 张）

![TP](docs/images/yolo_seg_eval_cases/tp_case_01.png)

YOLO 成功检出缺陷，预测 mask 与 GT mask 重合良好。

### TN 案例（894 张）

![TN](docs/images/yolo_seg_eval_cases/tn_case_01.png)

OK 样本正确识别，无任何预测 mask。

### FN 案例（44 张）

![FN](docs/images/yolo_seg_eval_cases/fn_case_01.png)

仍有 44 张 NG 漏检。FN 的 GT 平均面积 5,242 > TP 的 3,876，说明不是缺陷太小的问题。

### FP 案例

默认 conf=0.25 下 **FP=0**。但不能因此断定工业现场一定不会误检——样本规模有限，且降低 conf 至 0.05 时出现 9 个 FP。

## 6. 当前 YOLO 模型局限

- 仅训练 3 epochs（smoke training），远未收敛
- CPU 训练速度有限，训练轮数不足
- 仍有 44 张 NG 漏检（Recall=60%）
- Mask 边界质量有提升空间（IoU=0.553）
- 仅使用了 yolov8n-seg（最小模型），未对比更大模型
- confidence=0.25 时 FP=0 可能因模型过于保守

## 7. 后续 V2.3 计划

- epochs=30 正式训练
- 对比 yolov8n-seg vs yolov8s-seg
- 记录完整训练曲线
- 重新评估置信度实验
- OpenCV CLAHE vs YOLO 最终对比报告
