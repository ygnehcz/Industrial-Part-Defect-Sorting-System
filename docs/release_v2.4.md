# V2.4 版本说明：YOLO 模型选择与错误案例分析

## 1. 版本定位

V2.4 不训练新模型，而是对 V2.2（3ep）和 V2.3（10ep）进行系统对比，找出真正的最佳模型和 confidence。

## 2. 模型选择实验

6 个 confidence × 2 个模型 = 12 组对比。

## 3. 最佳模型

**V2.2 yolov8n-seg 3ep, conf=0.05**
- F1=0.833, Precision=90.4%, Recall=77.3%
- 全面优于 10ep 版本

## 4. 当前结论

- 3 epochs 是本机 CPU 训练的最佳选择
- 10 epochs 带来过拟合，FP 增加，F1 下降
- 不推荐继续 CPU 训练更多 epoch

## 5. 下一步建议

- GPU 训练 yolov8s-seg（50+ epochs）
- 数据增强 + 学习率调度优化
