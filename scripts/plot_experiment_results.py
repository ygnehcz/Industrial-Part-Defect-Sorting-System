"""实验结果可视化脚本

读取已有 CSV 实验数据，生成对比图表。
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHARTS_DIR = Path("docs/images/experiment_charts")
REPORTS_DIR = Path("outputs/real_dataset/reports")

CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_v1_1_vs_v1_2():
    """V1.1 vs V1.2 指标对比"""
    categories = ["V1.1 Baseline", "V1.1 Best Recall", "V1.2 CLAHE Best F1"]
    precision = [1.0, 0.114, 0.823]
    recall = [0.045, 1.0, 0.591]
    f1 = [0.087, 0.205, 0.688]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, precision, width, label="Precision", color="#2196F3")
    bars2 = ax.bar(x, recall, width, label="Recall", color="#4CAF50")
    bars3 = ax.bar(x + width, f1, width, label="F1 Score", color="#FF9800")

    ax.set_ylabel("Score")
    ax.set_title("V1.1 vs V1.2: Precision / Recall / F1 Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 1.1)

    for bar in bars1 + bars2 + bars3:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "v1_1_vs_v1_2_metrics.png", dpi=150)
    plt.close()
    print("Generated: v1_1_vs_v1_2_metrics.png")


def plot_enhanced_cv_top10():
    """Enhanced CV 实验 Top 10 F1"""
    data = load_csv(REPORTS_DIR / "enhanced_cv_experiment.csv")
    sorted_data = sorted(data, key=lambda x: float(x["f1_score"]), reverse=True)[:10]

    labels = [f"{r['method']}\n{r['parameters'][:30]}" for r in sorted_data]
    f1_scores = [float(r["f1_score"]) for r in sorted_data]
    recalls = [float(r["recall"]) for r in sorted_data]
    precisions = [float(r["precision"]) for r in sorted_data]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.25

    ax.bar(x - width, f1_scores, width, label="F1", color="#FF9800")
    ax.bar(x, recalls, width, label="Recall", color="#4CAF50")
    ax.bar(x + width, precisions, width, label="Precision", color="#2196F3")

    ax.set_ylabel("Score")
    ax.set_title("Enhanced CV Methods: Top 10 by F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "enhanced_cv_methods_f1_top10.png", dpi=150)
    plt.close()
    print("Generated: enhanced_cv_methods_f1_top10.png")


def plot_precision_recall_tradeoff():
    """Precision vs Recall 散点图"""
    data = load_csv(REPORTS_DIR / "enhanced_cv_experiment.csv")

    methods = set(r["method"] for r in data)
    colors = {"clahe_fixed": "#2196F3", "clahe_adaptive": "#03A9F4",
              "tophat_fixed": "#4CAF50", "blackhat_fixed": "#9C27B0",
              "sobel_fixed": "#FF9800", "canny_filter": "#F44336"}
    markers = {"clahe_fixed": "o", "clahe_adaptive": "s",
               "tophat_fixed": "^", "blackhat_fixed": "v",
               "sobel_fixed": "D", "canny_filter": "x"}

    fig, ax = plt.subplots(figsize=(8, 7))

    for method in methods:
        subset = [r for r in data if r["method"] == method]
        prec = [float(r["precision"]) for r in subset]
        rec = [float(r["recall"]) for r in subset]
        ax.scatter(rec, prec, c=colors.get(method, "gray"),
                   marker=markers.get(method, "o"), label=method.replace("_", " "),
                   alpha=0.7, s=30)

    # V1.1 baseline and best recall markers
    ax.scatter([0.045], [1.0], c="red", marker="*", s=200, zorder=5, label="V1.1 Baseline")
    ax.scatter([1.0], [0.114], c="darkred", marker="*", s=200, zorder=5, label="V1.1 Best Rec")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall: All Enhanced CV Experiments")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "precision_recall_tradeoff.png", dpi=150)
    plt.close()
    print("Generated: precision_recall_tradeoff.png")


def plot_confusion_matrix():
    """V1.2 最佳方法混淆矩阵"""
    tp, tn, fp, fn = 65, 880, 14, 45
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted OK", "Predicted NG"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual OK", "Actual NG"])

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > 400 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16, color=color, fontweight="bold")

    ax.set_title("V1.2 CLAHE Confusion Matrix\n(TP=65, TN=880, FP=14, FN=45)")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "confusion_matrix_v1_2.png", dpi=150)
    plt.close()
    print("Generated: confusion_matrix_v1_2.png")


def main():
    print("Generating experiment visualization charts...")
    plot_v1_1_vs_v1_2()
    plot_enhanced_cv_top10()
    plot_precision_recall_tradeoff()
    plot_confusion_matrix()
    print(f"\nAll charts saved to: {CHARTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
