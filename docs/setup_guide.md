# 环境搭建指南

## 环境要求

- Python 3.9+
- Windows / Linux / macOS
- 4GB+ RAM（CPU 训练）
- 可选：NVIDIA GPU（加速 YOLO 训练）

## 安装依赖

```bash
pip install opencv-python numpy pandas matplotlib
# YOLO 可选（demo 和训练需要）
pip install ultralytics
```

## 准备数据

数据集已内置在项目中。如需使用完整原始数据：

```bash
# 原始数据路径
D:\Industrial-Surface-Defect-Inspection\data\raw\
```

生成 YOLO segmentation 数据集：

```bash
python scripts/prepare_yolo_seg_dataset.py
```

## 主要脚本

| 脚本 | 用途 |
|------|------|
| `main.py` | OpenCV baseline 检测 |
| `demo_run.py` | YOLO 演示 |
| `scripts/run_full_testset.py` | 完整 test 集检测 |
| `scripts/run_enhanced_cv_experiment.py` | 增强方法实验 |
| `scripts/train_yolo_seg.py` | YOLO 训练 |
| `scripts/evaluate_yolo_seg.py` | YOLO 评价 |
| `scripts/check_project_ready.py` | 项目就绪检查 |

## 如何复现实验

```bash
# 1. 就绪检查
python scripts/check_project_ready.py

# 2. OpenCV baseline
python main.py
python scripts/run_full_testset.py

# 3. CLAHE 增强实验
python scripts/run_enhanced_cv_experiment.py

# 4. YOLO 数据集准备
python scripts/prepare_yolo_seg_dataset.py

# 5. YOLO 训练（需 ultralytics）
python scripts/train_yolo_seg.py

# 6. YOLO 评价
python scripts/evaluate_yolo_seg.py
```

## 注意事项

- 首次运行 YOLO 训练会自动下载 yolov8n-seg.pt（约 6MB）
- CPU 训练 3 epochs 约需 30-40 分钟
- 训练输出在 `outputs/yolo_seg/runs/` 下
- 所有 CSV 实验报表在 `outputs/*/reports/` 下
