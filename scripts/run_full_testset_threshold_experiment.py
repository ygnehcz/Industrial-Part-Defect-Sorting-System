"""完整 test 集阈值方法实验

运行方式（在项目根目录执行）：
    python scripts/run_full_testset_threshold_experiment.py
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gt_mask_utils import find_image_mask_pairs, summarize_dataset_by_mask
from src.threshold_experiment import run_all_experiments, find_best_method

TEST_DATASET_DIR = r"D:\Industrial-Surface-Defect-Inspection\data\raw\test"
REAL_REPORTS_DIR = "outputs/real_dataset/reports"


def main():
    print("=" * 60)
    print("完整 test 集阈值方法实验")
    print("=" * 60)

    summary = summarize_dataset_by_mask(TEST_DATASET_DIR)
    print(f"\n数据集：{summary['total_count']} 对 (OK={summary['ok_count']}, NG={summary['ng_count']})")

    pairs = find_image_mask_pairs(TEST_DATASET_DIR)
    print(f"匹配到 {len(pairs)} 对图像-mask")

    experiments = run_all_experiments(pairs)

    reports_dir = Path(REAL_REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / "full_testset_threshold_experiment.csv"
    fields = ["method", "gray_threshold", "area_threshold", "kernel_size",
              "tp", "tn", "fp", "fn", "total_count", "accuracy", "precision", "recall"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(experiments)

    best = find_best_method(experiments)

    print(f"\n实验完成，共 {len(experiments)} 组。报表：{csv_path}")
    print(f"\n最佳方法：{best['method']}, gray={best['gray_threshold']}, area={best['area_threshold']}, kernel={best['kernel_size']}")
    print(f"TP={best['tp']}, TN={best['tn']}, FP={best['fp']}, FN={best['fn']}")
    print(f"Accuracy={best['accuracy']*100:.1f}%, Precision={best['precision']*100:.1f}%, Recall={best['recall']*100:.1f}%")

    # Show top 5 by recall
    sorted_by_recall = sorted(experiments, key=lambda x: (-x["recall"], -x["precision"]))
    print(f"\nTop 5 (按 recall 降序)：")
    for i, e in enumerate(sorted_by_recall[:5]):
        print(f"  {i+1}. {e['method']:8s} gray={str(e['gray_threshold']):6s} area={str(e['area_threshold']):4s} kernel={e['kernel_size']}  "
              f"TP={e['tp']:3d} TN={e['tn']:3d} FP={e['fp']:3d} FN={e['fn']:3d}  "
              f"Acc={e['accuracy']*100:.1f}% Prec={e['precision']*100:.1f}% Rec={e['recall']*100:.1f}%")

    # Otsu and adaptive best
    otsu_best = max((e for e in experiments if e["method"] == "otsu"), key=lambda x: (x["recall"], x["precision"]))
    adaptive_best = max((e for e in experiments if e["method"] == "adaptive"), key=lambda x: (x["recall"], x["precision"]))
    print(f"\nOtsu 最佳：area={otsu_best['area_threshold']}, TP={otsu_best['tp']}, FP={otsu_best['fp']}, FN={otsu_best['fn']}, Recall={otsu_best['recall']*100:.1f}%")
    print(f"Adaptive 最佳：area={adaptive_best['area_threshold']}, TP={adaptive_best['tp']}, FP={adaptive_best['fp']}, FN={adaptive_best['fn']}, Recall={adaptive_best['recall']*100:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()
