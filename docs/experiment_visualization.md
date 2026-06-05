# 实验结果可视化分析

## 为什么需要可视化

V1.1 和 V1.2 生成了大量 CSV 实验数据（100+ 组阈值实验 + 54 组增强实验），仅通过表格难以直观对比方法优劣。可视化图表帮助快速理解：

- 各版本之间的指标差异
- 不同方法在 Precision-Recall 平面上的分布
- 混淆矩阵的实际分布

## V1.1 与 V1.2 指标对比

![V1.1 vs V1.2](docs/images/experiment_charts/v1_1_vs_v1_2_metrics.png)

- V1.1 基线：Precision=100% 但 Recall 仅 4.5%，F1=0.087
- V1.1 最佳 Recall：Recall=100% 但 Precision 仅 11.4%，F1=0.205
- **V1.2 CLAHE**：Recall=59.1%，Precision=82.3%，**F1=0.688**

CLAHE 在两者之间取得了最好的平衡，F1 是 V1.1 基线的 7.9 倍。

## Top 10 F1 方法

![Top 10 F1](docs/images/experiment_charts/enhanced_cv_methods_f1_top10.png)

- CLAHE + fixed threshold 占据前 4 名，是压倒性的最佳方法
- Sobel 边缘检测排第 5，但 F1 显著低于 CLAHE
- Top-hat、Black-hat、Canny 未进入前 5

## Precision-Recall 取舍

![Precision-Recall Tradeoff](docs/images/experiment_charts/precision_recall_tradeoff.png)

散点图展示了所有 54 组增强实验在 Precision-Recall 平面上的分布：

- CLAHE（蓝色圆点）集中在右上区域——同时具有较高的 Precision 和 Recall
- 五角星标注了 V1.1 基线（左上）和 V1.1 最佳 Recall（右下）——两者处于极端位置
- 其他方法（Top-hat、Black-hat、Sobel、Canny）分布在左下或左中区域

理想的方法应位于**右上角**（高 Precision + 高 Recall），CLAHE 是目前最接近这个目标的。

## V1.2 混淆矩阵

![Confusion Matrix](docs/images/experiment_charts/confusion_matrix_v1_2.png)

- TN=880（89.5%）：绝大多数 OK 样本正确识别
- TP=65（59.1%）：过半 NG 样本被检出
- FP=14（1.4%）：少量 OK 被误判
- FN=45（40.9%）：仍有近半缺陷漏检

## 为什么选择 CLAHE

1. **F1 最高**（0.688）：在 Precision 和 Recall 之间取得最佳平衡
2. **Top 4 全是 CLAHE**：方法稳定性好，参数调优空间可控
3. **实现简单**：仅增加一行 `cv2.createCLAHE()` 调用
4. **原理清晰**：局部对比度增强，对光照不均和低对比度场景有效
5. **FP 可控**：14/894 = 1.6% 误报率，工业场景可接受
