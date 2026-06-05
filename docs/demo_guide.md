# 项目演示指南

## 推荐演示顺序

1. 检查项目就绪状态
2. 运行 OpenCV baseline
3. 运行 YOLO demo
4. 展示结果图和指标

## 如何运行 OpenCV baseline

```bash
python main.py
```

输出：模拟图片检测结果（6 张样本，100% 准确率）。

## 如何运行真实数据检测

```bash
python scripts/run_full_testset.py
```

输出：1004 张 val 集 OpenCV CLAHE 检测结果，TP/TN/FP/FN 统计。

## 如何运行 YOLO demo

```bash
python demo_run.py
```

输出：8 张演示图片的 YOLO segmentation 推理结果，保存到 `outputs/demo/`。

## 如何查看结果图

- `outputs/demo/` — YOLO 演示结果（含 mask 叠加图）
- `outputs/real_dataset/full_testset_marked/` — OpenCV 标注结果
- `docs/images/yolo_seg_eval_cases/` — YOLO 4 联评价图
- `docs/images/experiment_charts/` — 实验对比图表

## 如何向面试官讲解

1. **开场**：这是一个工业表面缺陷检测项目，3335 张真实样本
2. **OpenCV**：展示 CLAHE 增强前/后的指标对比（F1 0.087→0.688）
3. **YOLO**：展示 YOLO 3epoch 结果（F1=0.833）
4. **对比**：OpenCV vs YOLO 的 Precision-Recall 图
5. **分析**：25 张 FN 的案例分析，Precision-Recall 权衡
6. **总结**：从传统方法到深度学习的完整演进

## 常见问题

### ultralytics 未安装怎么办

```bash
pip install ultralytics
```

### best.pt 不存在怎么办

从 GitHub Releases 下载，或运行 `python scripts/train_yolo_seg.py` 重新训练。

### outputs 被 .gitignore 忽略是否正常

正常。训练输出和检测结果不应提交到 Git，可本地运行重新生成。

### 为什么不继续使用 10 epochs 模型

3ep F1=0.833 > 10ep F1=0.808。10ep 在 CPU 小 batch 下出现了过拟合。
