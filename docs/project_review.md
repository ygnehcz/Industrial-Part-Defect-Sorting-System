# 工业零件表面缺陷检测与分拣模拟系统项目复盘

## 1. 项目目标

本项目面向工业视觉检测场景，目标是模拟工业零件表面缺陷检测流程，完成从图像输入、缺陷识别、OK/NG 判断、结果保存到评价分析的完整闭环。项目采用传统机器视觉方法实现，适合作为工业视觉入门学习和简历作品集展示。

## 2. 技术路线

当前采用传统机器视觉路线：

- 编程语言：Python
- 图像处理：OpenCV
- 预处理：灰度化 + 二值化阈值分割
- 缺陷分析：轮廓检测 + 缺陷面积统计
- 标注：矩形框绘制 OK/NG 判定结果
- 报表：CSV 格式保存检测明细与汇总
- 评价：混淆矩阵 + Accuracy / Precision / Recall
- 实验：多阈值参数对比，绘制指标曲线
- 版本管理：Git + GitHub

## 3. 当前实现功能

- 批量生成测试图片（`scripts/create_batch_sample_images.py`）
- 批量读取图片
- 图像灰度化
- 二值化阈值分割
- 缺陷面积计算
- 缺陷轮廓检测
- 缺陷框标注（marked 图和 binary 图）
- 根据面积阈值判断 OK / NG
- 根据文件名提取真实标签
- 统计 TP / TN / FP / FN
- 计算 Accuracy、Precision、Recall
- 保存检测明细 CSV（`batch_detection_report.csv`）
- 保存汇总评价 CSV（`batch_summary_metrics.csv`）
- 保存阈值实验 CSV（`threshold_experiment_report.csv`）
- 绘制阈值指标曲线图（`threshold_metrics.png`）
- README 中展示检测效果图

## 4. 核心模块说明

| 模块 | 说明 |
|------|------|
| `main.py` | 主流程入口，组织批量检测、评价和报表保存 |
| `config.py` | 保存阈值、路径等配置参数 |
| `src/image_processor.py` | 图像读取、灰度化、二值化、面积计算、轮廓检测和标注 |
| `src/judge.py` | 根据缺陷面积判断 OK / NG |
| `src/evaluator.py` | 计算 TP / TN / FP / FN 和评价指标 |
| `src/report.py` | 保存 CSV 报表 |
| `src/visualizer.py` | 绘制阈值实验曲线 |
| `scripts/create_batch_sample_images.py` | 生成测试样例图片（含 OK 和 NG 样本） |

## 5. 输出结果

| 路径 | 说明 |
|------|------|
| `outputs/images/` | 二值图（binary）和标注图（marked） |
| `outputs/reports/batch_detection_report.csv` | 批量检测明细报表 |
| `outputs/reports/batch_summary_metrics.csv` | 汇总评价指标报表 |
| `outputs/reports/threshold_experiment_report.csv` | 多阈值实验对比报表 |
| `outputs/reports/threshold_metrics.png` | 阈值参数-指标曲线图 |
| `docs/images/` | README 展示用效果图（OK / NG / 多缺陷样例） |

## 6. 当前项目亮点

- 完成了从图像输入到结果评价的完整检测闭环，流程清晰、结构完整
- 支持批量检测，而非单张图片处理，更贴近产线实际场景
- 输出结构化 CSV 报表，便于后续数据分析与追溯
- 引入混淆矩阵和 Accuracy / Precision / Recall 等多维评价指标
- 设计多阈值参数实验，体现参数调优思路和工程严谨性
- 项目结构模块化，按功能拆分 `src/`、`scripts/`、`outputs/`、`docs/`
- 具备 GitHub 项目展示能力，包含 README 效果图、完整代码和文档

## 7. 当前局限

- 当前样本主要为程序生成的模拟图片，非真实工业数据集
- 检测方法基于简单固定阈值，对复杂光照和纹理场景鲁棒性有限
- 缺陷类型较为单一（以面积大小为主）
- 尚无 GUI 交互界面
- 尚未接入相机进行实时检测
- 尚未与 YOLO 等深度学习方法进行对比

## 8. 后续改进方向

- 引入真实工业缺陷图片数据集，增强实用性
- 增加形态学开闭运算，减少噪声干扰
- 增加 HSV 色彩空间、自适应阈值等预处理方法
- 开发简单 GUI 界面，降低使用门槛
- 接入工业相机，实现实时视频流检测
- 增加检测结果导出和历史记录管理功能
- 引入 YOLO 等深度学习目标检测方案进行对比
- 进一步整理为简历作品集项目，补充技术博客或演示视频

## 9. 面试讲法

这个项目是我为了系统学习工业机器视觉流程而做的一个传统 OpenCV 检测项目。整体思路是从批量读取图片开始，经过灰度化和阈值分割做预处理，再通过轮廓检测提取缺陷区域，根据面积阈值判断 OK / NG，最后输出 CSV 报表和评价指标。

在项目里，我重点练习了几个方面：一是图像处理的基本流程，包括灰度化、二值化、轮廓提取和标注；二是参数调试的思路，比如阈值选多大合适，我做了多阈值实验，对比 Accuracy、Precision、Recall 的变化曲线来确定推荐值；三是工程化习惯，把代码按功能拆成 image_processor、judge、evaluator、report 等模块，用 Git 做版本管理，写 README 和复盘文档。

当前这个版本还是传统规则版，依赖于固定阈值。后续我计划引入真实工业数据、加上 GUI 界面，再和 YOLO 这类深度学习方法做个对比，这样能更完整地体现从传统方法到深度学习的技术演进过程。
