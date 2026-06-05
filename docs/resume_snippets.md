# 简历素材

## 一行版

基于 OpenCV + YOLOv8-seg 的工业表面缺陷检测系统，在 3335 张真实样本上完成从传统视觉到深度学习的完整演进验证。

## 三行版（适合项目经历）

**工业零件表面缺陷检测系统**
- 使用 Python/OpenCV 搭建传统视觉检测流水线（CLAHE 增强 + 形态学 + 阈值分割），在 1004 张真实工业样本上验证，F1=0.688
- 将像素级 GT mask 转换为 YOLO segmentation 格式，训练 YOLOv8n-seg 模型，F1 提升至 0.833（+21% vs OpenCV）
- 完成 100+ 组参数实验、Precision-Recall 权衡分析、模型选择对比和完整项目文档

## STAR 面试讲述版

**S（情境）**：我想系统学习工业缺陷检测的完整流程，从传统视觉到深度学习。

**T（任务）**：选了一个 KolektorSDD2 风格的工业表面缺陷数据集（3335 对像素级标注图像），目标是从 OpenCV baseline 开始，逐步演进到 YOLO segmentation。

**A（行动）**：
- 先用固定阈值做 baseline，真实数据上 Recall 仅 4.5%
- 引入 CLAHE 增强，F1 从 0.087 提升到 0.688
- 将 GT mask 转为 YOLO segmentation 格式，训练 yolov8n-seg
- 做了 100+ 组 OpenCV 参数实验、50+ 组增强方法对比、confidence sweep 和 3ep vs 10ep 模型选择
- 完整的 Git 版本管理（14 个 tag）和文档体系

**R（结果）**：YOLO 3ep 取得 F1=0.833（Precision=90.4%, Recall=77.3%），24 个 FP 和 25 个 FN。所有实验数据和图表均可在 GitHub 查看。

## 技术栈关键词

Python, OpenCV, NumPy, Matplotlib, CLAHE, YOLOv8-seg, ultralytics, tkinter, Git, CSV

## 项目亮点

- 不是"调个 API 跑一下"，而是从零搭建全流程检测流水线
- 传统方法 → 深度学习的完整演进路线，每一步有数据支撑
- 100+ 组实验证明 CLAHE 是传统方法中最优选择
- YOLO 仅 3 epoch 即超越 CLAHE 21%
- 真实数据、真实指标、真实的失败案例（FN=25, FP=9）——不夸大

## 项目局限的诚实表述

- CPU 训练仅 3 epochs，YOLO 的潜力远未完全释放
- 数据集 NG 仅 356 张，类别不平衡
- Recall 77.3% 有提升空间
- 未部署到实际产线验证

## 适合岗位

机器视觉应用工程师 / 图像算法助理工程师 / AI 应用开发 / 缺陷检测工程师
