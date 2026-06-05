"""真实工业缺陷数据集 OpenCV 检测运行脚本

运行方式（在项目根目录执行）：
    python scripts/run_real_dataset.py
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.real_dataset_detector import (
    run_real_dataset_detection,
    compute_metrics,
    REAL_REPORTS_DIR
)


def main():
    print("=" * 50)
    print("真实工业缺陷数据集 OpenCV 检测")
    print("=" * 50)

    results = run_real_dataset_detection()
    metrics = compute_metrics(results)

    reports_dir = Path(REAL_REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)

    detail_path = reports_dir / "real_detection_report.csv"
    summary_path = reports_dir / "real_summary_metrics.csv"

    detail_fields = [
        "image_name", "image_path", "true_status", "pred_status",
        "defect_area", "defect_count", "result_type",
        "marked_image_path", "processed_image_path"
    ]
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(results)

    summary_fields = [
        "tp", "tn", "fp", "fn", "total_count",
        "accuracy", "precision", "recall"
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow(metrics)

    ok_total = sum(1 for r in results if r["true_status"] == "OK")
    ng_total = sum(1 for r in results if r["true_status"] == "NG")

    print(f"\n总样本数：{len(results)}  (OK={ok_total}, NG={ng_total})")
    print(f"TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
    print(f"Accuracy  = {metrics['accuracy'] * 100:.1f}%")
    print(f"Precision = {metrics['precision'] * 100:.1f}%")
    print(f"Recall    = {metrics['recall'] * 100:.1f}%")

    fp_samples = [r for r in results if r["result_type"] == "FP"]
    fn_samples = [r for r in results if r["result_type"] == "FN"]

    if fp_samples:
        print(f"\n误检 (FP) 样本：{len(fp_samples)} 张")
        for r in fp_samples:
            print(f"  {r['image_name']} — 缺陷面积={r['defect_area']}")
    else:
        print(f"\n误检 (FP)：0 张（当前子集未出现误检）")

    if fn_samples:
        print(f"\n漏检 (FN) 样本：{len(fn_samples)} 张")
        for r in fn_samples:
            print(f"  {r['image_name']} — 缺陷面积={r['defect_area']}")
    else:
        print(f"\n漏检 (FN)：0 张（当前子集未出现漏检）")

    print(f"\n检测明细报表：{detail_path}")
    print(f"汇总指标报表：{summary_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
