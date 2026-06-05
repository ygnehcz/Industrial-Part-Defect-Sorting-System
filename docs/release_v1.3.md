# V1.3 版本说明：实验结果可视化与展示增强

## 1. 版本定位

V1.3 不增加新的检测方法，而是将 V1.1 和 V1.2 的实验数据进行可视化，提升项目的展示能力和 GitHub 可读性。

## 2. 新增图表

| 图表 | 说明 |
|------|------|
| `v1_1_vs_v1_2_metrics.png` | V1.1 vs V1.2 Precision/Recall/F1 柱状对比 |
| `enhanced_cv_methods_f1_top10.png` | 增强实验 Top 10 F1 方法 |
| `precision_recall_tradeoff.png` | 所有方法的 Precision-Recall 散点图 |
| `confusion_matrix_v1_2.png` | V1.2 最佳方法混淆矩阵 |

## 3. 项目展示增强点

- 图表使实验结果一目了然，无需阅读 CSV
- Precision-Recall 散点图清晰展示了传统方法的性能上限
- 混淆矩阵直观反映了各分类的实际分布
- 所有图表均可直接用于简历、汇报和面试展示

## 4. 后续 V2.0 计划

- YOLO 缺陷检测训练与对比
- 在同一图表中加入 YOLO 的 Precision-Recall 点
- 像素级 IoU/Dice 分割评估
- 实时检测性能对比
