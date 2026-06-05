# 工业零件表面缺陷检测与分拣模拟系统

## 项目简介

本项目用于学习和实现工业零件表面缺陷检测与分拣模拟流程。

当前阶段使用 Python + OpenCV 实现了一个传统机器视觉检测流程：

1. 批量读取测试图片
2. 灰度化与阈值分割
3. 计算缺陷区域面积
4. 查找缺陷轮廓并绘制标注框
5. 根据缺陷面积判断 OK / NG
6. 根据文件名提取真实标签
7. 统计 TP / TN / FP / FN
8. 计算 Accuracy、Precision、Recall
9. 保存检测明细报表、汇总评价报表和阈值实验曲线图


## 效果展示

### OK 合格样例

![OK 合格样例](docs/images/ok_sample_marked.jpg)

### NG 单缺陷样例

![NG 单缺陷样例](docs/images/ng_sample_marked.jpg)

### NG 多缺陷样例

![NG 多缺陷样例](docs/images/multi_defect_marked.jpg)


## V0.2 更新说明

- 增加形态学开运算、闭运算预处理模块 (`src/morphology.py`)
- 缺陷面积统计和轮廓检测改为基于形态学处理后图像
- 每张图片额外保存形态学处理结果图到 `outputs/morphology/`
- CSV 报表新增 `morph_image_path` 字段
- 开运算用于去除小噪声点，闭运算用于填补缺陷区域断裂

## 当前项目结构

```text
Industrial-Part-Defect-Sorting-System
├─ data
│  └─ samples
├─ outputs
│  ├─ images
│  └─ reports
├─ scripts
│  └─ create_batch_sample_images.py
├─ src
│  ├─ image_processor.py
│  ├─ judge.py
│  ├─ evaluator.py
│  ├─ report.py
│  └─ visualizer.py
├─ config.py
├─ main.py
├─ requirements.txt
└─ README.md









