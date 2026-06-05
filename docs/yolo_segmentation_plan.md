# V2.0 YOLO 分割训练框架说明

## 1. 为什么选择 YOLO segmentation

当前数据集（KolektorSDD2 风格）提供像素级 `*_GT.png` mask，而不仅仅是边界框。选择 YOLO segmentation 而非 detection 的理由：

- mask 中的缺陷形状不规则，用 bounding box 会包含大量背景
- segmentation 可以提供像素级缺陷定位，评估更精细
- YOLOv8-seg 同时支持检测和分割，一份训练两份能力
- 后续可以用 mask IoU / Dice 做像素级评估，比框级 mAP 更适合此场景

## 2. 数据集转换流程

```
*_GT.png mask
  → 二值化
  → cv2.findContours 提取外轮廓
  → cv2.approxPolyDP 简化多边形
  → 坐标归一化到 0~1
  → 写入 YOLO segmentation txt 标签
```

- 过滤面积 < 5 像素的轮廓
- 空 mask（OK 样本）生成空 txt 文件
- 类别 ID：0，类别名：defect
- 保持原始 train/test 划分不变

## 3. 数据集结构

```
data/yolo_seg/
├── images/
│   ├── train/    (2331 张)
│   └── val/      (1004 张)
├── labels/
│   ├── train/    (2331 个 txt)
│   └── val/      (1004 个 txt)
└── data.yaml
```

| 划分 | 总数 | OK | NG |
|------|------|-----|-----|
| train | 2331 | 2085 | 246 |
| val | 1004 | 894 | 110 |
| **合计** | **3335** | **2979** | **356** |

## 4. 与 OpenCV 方法的关系

- **OpenCV CLAHE**（V1.2）：传统视觉 baseline，F1=0.688
- **YOLO segmentation**（V2.0）：深度学习扩展方向
- 后续 V2.1 计划在同一 val 集上对比两者的 Precision / Recall / F1
- YOLO 的优势：自动学习特征、对光照/纹理更鲁棒

## 5. 运行方式

```bash
# 1. 数据集转换
python scripts/prepare_yolo_seg_dataset.py

# 2. 数据集检查
python scripts/check_yolo_seg_dataset.py

# 3. 标注可视化
python scripts/visualize_yolo_seg_labels.py

# 4. 训练（需要 ultralytics）
python scripts/train_yolo_seg.py

# 5. 推理（需要训练好的模型）
python scripts/predict_yolo_seg.py
```

## 6. 当前 V2.0 局限

- V2.0 完成了数据转换、检查和训练框架搭建
- ultralytics 尚未在当前环境安装，暂未实际训练
- 真正效果需要训练完成后在 val 集上评估
- 后续 V2.1 应加入 YOLO 推理结果的 TP/TN/FP/FN 统计

## 7. 后续 V2.1 计划

- 安装 ultralytics 并训练 yolov8n-seg 模型
- 对 val 集进行推理
- 根据 GT mask 计算分割指标（IoU、Dice）
- 统计 OK/NG 分类 Accuracy、Precision、Recall、F1
- 与 OpenCV CLAHE baseline 进行量化对比
- 绘制 OpenCV vs YOLO 的 Precision-Recall 对比曲线
