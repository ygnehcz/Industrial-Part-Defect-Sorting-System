# V2.0 版本说明：YOLO 分割训练框架

## 1. 版本定位

V2.0 是项目从传统视觉方法迈向深度学习的转折版本。基于 KolektorSDD2 风格的像素级 GT mask，将数据集转换为 YOLO segmentation 格式，建立完整的训练-推理框架。

## 2. 相比 V1.3 的提升

- 从传统 OpenCV 方法扩展到 YOLO 深度学习框架
- 利用 GT mask 的像素级信息做 segmentation，而非仅 OK/NG 二分类
- 3335 对训练+验证数据，train/val 划分与原数据集一致
- 完整的工程化脚本链：转换 → 检查 → 可视化 → 训练 → 推理

## 3. 新增功能

| 文件 | 说明 |
|------|------|
| `scripts/prepare_yolo_seg_dataset.py` | GT mask → YOLO segmentation polygon 转换 |
| `scripts/check_yolo_seg_dataset.py` | 数据集完整性检查 |
| `scripts/visualize_yolo_seg_labels.py` | 标注可视化 |
| `scripts/train_yolo_seg.py` | YOLOv8-seg 训练脚本 |
| `scripts/predict_yolo_seg.py` | YOLOv8-seg 推理脚本 |
| `data/yolo_seg/` | YOLO segmentation 数据集 |
| `docs/yolo_segmentation_plan.md` | YOLO 分割训练框架说明 |
| `docs/images/yolo_seg_samples/` | 标注可视化样例 |

## 4. 数据集统计

| 划分 | 总数 | OK | NG |
|------|------|-----|-----|
| train | 2331 | 2085 | 246 |
| val | 1004 | 894 | 110 |
| **合计** | **3335** | **2979** | **356** |

## 5. ultralytics 状态

- ultralytics 当前**未安装**
- 数据转换、检查、可视化均已完成且通过
- 训练脚本已就绪，安装后即可运行

## 6. 是否完成真实训练

**否**。V2.0 完成了数据集准备和训练框架搭建。实际训练需要：

```bash
pip install ultralytics
python scripts/train_yolo_seg.py
```

## 7. 后续 V2.1 计划

- 安装 ultralytics 并训练 yolov8n-seg
- val 集推理与指标评估
- OpenCV CLAHE vs YOLO 量化对比
- 像素级 IoU/Dice 分割评估
