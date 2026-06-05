"""YOLO 训练对比图表"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("docs/images/yolo_training_charts")
OUT.mkdir(parents=True, exist_ok=True)

# Data
labels = ["V1.2 CLAHE", "V2.1 YOLO 3ep", "V2.2 YOLO 3ep\n(conf=0.05)", "V2.3 YOLO 10ep\n(conf=0.05)"]
f1_scores = [0.688, 0.750, 0.833, 0.808]
precisions = [0.823, 1.0, 0.904, 0.835]
recalls = [0.591, 0.600, 0.773, 0.782]

# F1 comparison
fig, ax = plt.subplots(figsize=(9,5))
x = np.arange(len(labels))
colors = ["#2196F3","#4CAF50","#FF9800","#F44336"]
ax.bar(x, f1_scores, color=colors, width=0.5)
for i,v in enumerate(f1_scores):
    ax.text(i, v+0.01, f"{v:.3f}", ha="center", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("F1 Score"); ax.set_title("F1 Score Comparison Across Versions")
ax.set_ylim(0, 1.0)
plt.tight_layout(); plt.savefig(OUT/"yolo_f1_comparison.png", dpi=150); plt.close()

# Precision-Recall comparison
fig, ax = plt.subplots(figsize=(8,6))
for i, (lbl, prec, rec) in enumerate(zip(labels, precisions, recalls)):
    ax.scatter(rec, prec, c=colors[i], s=150, zorder=5)
    ax.annotate(lbl.replace("\n"," "), (rec, prec), textcoords="offset points", xytext=(8,5), fontsize=8)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision vs Recall Across Versions")
ax.set_xlim(0.5, 1.05); ax.set_ylim(0.5, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(OUT/"yolo_precision_recall_comparison.png", dpi=150); plt.close()

# IoU/Dice comparison
iou_vals = [0, 0.553, 0.553, 0.474]
dice_vals = [0, 0.698, 0.698, 0.624]
fig, ax = plt.subplots(figsize=(8,5))
x2 = np.arange(len(labels)); w = 0.3
ax.bar(x2-w/2, iou_vals, w, label="mean IoU (TP)", color="#2196F3")
ax.bar(x2+w/2, dice_vals, w, label="mean Dice (TP)", color="#FF9800")
for i,(iou,dice) in enumerate(zip(iou_vals,dice_vals)):
    if iou>0: ax.text(i-w/2, iou+0.01, f"{iou:.3f}", ha="center", fontsize=8)
    if dice>0: ax.text(i+w/2, dice+0.01, f"{dice:.3f}", ha="center", fontsize=8)
ax.set_xticks(x2); ax.set_xticklabels(labels)
ax.set_ylabel("Score"); ax.set_title("IoU / Dice Comparison (TP images)")
ax.legend(); ax.set_ylim(0, 0.9)
plt.tight_layout(); plt.savefig(OUT/"yolo_iou_dice_comparison.png", dpi=150); plt.close()
print(f"Charts saved to {OUT}")
