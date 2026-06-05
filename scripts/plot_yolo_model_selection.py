"""模型选择图表"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path

OUT = Path("docs/images/yolo_model_selection_charts")
OUT.mkdir(parents=True, exist_ok=True)

with open("outputs/yolo_seg/reports/yolo_model_comparison.csv") as f:
    rows = list(csv.DictReader(f))

# F1 vs conf curves
fig, ax = plt.subplots(figsize=(8,5))
for model_name in ["V2.2_3ep","V2.3_10ep"]:
    subset = [r for r in rows if r["model"]==model_name]
    confs = [float(r["conf"]) for r in subset]
    f1s = [float(r["f1_score"]) for r in subset]
    ax.plot(confs, f1s, marker="o", label=model_name, linewidth=2)
ax.set_xlabel("Confidence Threshold"); ax.set_ylabel("F1 Score")
ax.set_title("F1 Score vs Confidence Threshold")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(OUT/"model_conf_f1_curve.png", dpi=150); plt.close()

# Precision-Recall curves
fig, ax = plt.subplots(figsize=(8,6))
for model_name in ["V2.2_3ep","V2.3_10ep"]:
    subset = [r for r in rows if r["model"]==model_name]
    precs = [float(r["precision"]) for r in subset]
    recs = [float(r["recall"]) for r in subset]
    ax.plot(recs, precs, marker="s", label=model_name, linewidth=2)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xlim(0.3, 1.05); ax.set_ylim(0.3, 1.05)
plt.tight_layout(); plt.savefig(OUT/"model_conf_precision_recall_curve.png", dpi=150); plt.close()

# Best model confusion matrix
tp,tn,fp,fn=85,885,9,25
cm = np.array([[tn,fp],[fn,tp]])
fig, ax = plt.subplots(figsize=(5,5))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=16,fontweight="bold",color="white" if cm[i,j]>400 else "black")
ax.set_xticks([0,1]); ax.set_xticklabels(["Pred OK","Pred NG"])
ax.set_yticks([0,1]); ax.set_yticklabels(["Actual OK","Actual NG"])
ax.set_title("V2.2 3ep (conf=0.05) Confusion Matrix\nF1=0.833")
plt.tight_layout(); plt.savefig(OUT/"best_model_confusion_matrix.png", dpi=150); plt.close()
print(f"Charts saved to {OUT}")
