# V1.1 版本说明：完整 test 集 mask 分析与阈值实验

## 1. 版本定位

V1.1 在 V1.0 的 60 张子集基础上，扩展到完整 test 集 1004 对图像，使用 GT mask 作为真实标签，并进行了系统性的阈值参数实验。

## 2. 相比 V1.0 的提升

- 从 60 张子集扩展到 1004 对完整 test 集
- 真实标签从文件夹名升级为 GT mask（`*_GT.png` 像素级标注）
- 新增 GT 缺陷面积和缺陷比例统计
- 新增 100 组阈值参数实验（fixed / Otsu / Adaptive）
- 新增 Precision-Recall 权衡分析
- 新增 TP/TN/FP/FN 三联对比图
- 结论更扎实：漏检主要原因是灰度对比度不足，而非缺陷面积小

## 3. 新增功能

| 文件 | 说明 |
|------|------|
| `src/gt_mask_utils.py` | GT mask 读取、配对、统计工具 |
| `src/threshold_experiment.py` | 阈值参数实验模块 |
| `scripts/run_full_testset.py` | 完整 test 集检测脚本（基于 mask 标签） |
| `scripts/run_full_testset_threshold_experiment.py` | 完整 test 集阈值实验脚本 |
| `docs/full_testset_analysis.md` | 完整 test 集分析报告 |
| `docs/images/full_testset_cases/` | TP/TN/FN 三联对比图 |

## 4. 完整 test 集实验结果

### 基线（默认参数 gray=200, area=100, kernel=5）

| TP | TN | FP | FN | Accuracy | Precision | Recall |
|----|----|----|----|----------|-----------|--------|
| 5 | 894 | 0 | 105 | 89.5% | 100.0% | 4.5% |

### GT 缺陷面积分析

| | TP (n=5) | FN (n=105) |
|--|----------|------------|
| 平均 GT 缺陷面积 | 2,877.2 | 4,496.0 |
| 平均 GT 缺陷比例 | 1.97% | 3.11% |

### 最佳方法（Recall 最高）

| 方法 | gray | area | kernel | TP | TN | FP | FN | Recall | Precision |
|------|------|------|--------|----|----|----|----|--------|-----------|
| fixed | 60 | 20 | 3 | 110 | 41 | 853 | 0 | 100.0% | 11.4% |

## 5. 当前局限

- 固定阈值即使调到最佳（Recall 100%），Precision 也仅 11.4%
- Otsu/Adaptive 没有改善 Precision
- 核心瓶颈在缺陷分割阶段，不在分类阶段
- 所有方法都无法在 Precision 和 Recall 之间取得平衡

## 6. 后续 V2.0 计划

- YOLO 缺陷检测模型训练（解决分割瓶颈）
- U-Net 语义分割对比
- OpenCV vs YOLO 在完整 test 集上的量化对比
- 基于 GT mask 的像素级 IoU/Dice 评价
