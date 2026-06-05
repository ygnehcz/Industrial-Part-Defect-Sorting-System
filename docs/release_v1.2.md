# V1.2 版本说明：传统视觉增强与特征过滤改进实验

## 1. 版本定位

V1.2 在 V1.1 的阈值实验基础上，引入 CLAHE、Top-hat、Black-hat、Sobel、Canny 等传统视觉增强方法，探索是否能在 Precision 和 Recall 之间取得更好的平衡。

## 2. 相比 V1.1 的提升

| 指标 | V1.1 基线 | V1.1 最佳 Rec | **V1.2 最佳 F1** |
|------|-----------|---------------|-------------------|
| Recall | 4.5% | 100.0% | **59.1%** |
| Precision | 100.0% | 11.4% | **82.3%** |
| F1 | 0.087 | 0.205 | **0.688** |

CLAHE 增强后将 F1 从 0.087 提升到 0.688（7.9 倍），在 Recall 和 Precision 之间取得了显著更好的平衡。

## 3. 新增功能

| 文件 | 说明 |
|------|------|
| `src/enhanced_cv_methods.py` | CLAHE、Top-hat、Black-hat、Sobel、Canny、特征过滤 |
| `scripts/run_enhanced_cv_experiment.py` | 54 组增强实验 |
| `docs/enhanced_cv_analysis.md` | 详细分析报告 |
| `docs/images/enhanced_cv_cases/` | TP/FP/FN 三联对比图 |

## 4. 实验结果

### 最佳方法：CLAHE + fixed threshold

| TP | TN | FP | FN | Acc | Prec | Rec | F1 |
|----|----|----|----|-----|------|-----|-----|
| 65 | 880 | 14 | 45 | 94.1% | 82.3% | 59.1% | 0.688 |

### 方法排名（按 F1）

| # | 方法 | F1 | Rec | Prec |
|---|------|-----|-----|------|
| 1 | CLAHE fixed | 0.688 | 59.1% | 82.3% |
| 2 | CLAHE fixed | 0.584 | 42.7% | 92.2% |
| 3 | CLAHE fixed | 0.565 | 63.6% | 50.7% |
| 4 | CLAHE fixed | 0.503 | 33.6% | 100.0% |
| 5 | Sobel fixed | 0.480 | 44.5% | 52.1% |

## 5. 当前局限

- CLAHE 的 F1 天花板约 0.7，再调参改进空间有限
- Top-hat/Black-hat 效果不如 CLAHE，可能因缺陷类型不匹配
- 仍有 45/110 缺陷漏检，14/894 OK 误报
- 所有方法仍需人工调参

## 6. V2.0 深度学习计划

- YOLO 缺陷检测训练（解决 F1 天花板）
- U-Net 像素级分割对比
- 基于 GT mask 的 IoU/Dice 评估
- OpenCV vs YOLO 在完整 test 集上的量化对比
