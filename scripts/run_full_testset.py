"""完整 test 集 OpenCV 检测脚本（基于 GT mask 标签）

运行方式（在项目根目录执行）：
    python scripts/run_full_testset.py
"""

import sys
import csv
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    GRAY_THRESHOLD, THRESHOLD, MIN_CONTOUR_AREA,
    MORPH_KERNEL_SIZE, USE_OPEN_OPERATION, USE_CLOSE_OPERATION
)
from src.image_processor import (
    read_image, convert_to_gray, threshold_image,
    calculate_defect_area, find_defect_contours, draw_defect_boxes
)
from src.morphology import apply_morphology
from src.judge import judge_status
from src.gt_mask_utils import (
    find_image_mask_pairs, get_true_status_from_mask,
    calculate_gt_defect_area, calculate_gt_defect_ratio, summarize_dataset_by_mask
)

TEST_DATASET_DIR = r"D:\Industrial-Surface-Defect-Inspection\data\raw\test"
REAL_REPORTS_DIR = "outputs/real_dataset/reports"
MARKED_OUTPUT_DIR = "outputs/real_dataset/full_testset_marked"


def detect_and_evaluate(image_path, mask_path, marked_dir, gray_threshold, area_threshold):
    image = read_image(image_path)
    gray = convert_to_gray(image)
    binary = threshold_image(gray, gray_threshold)
    morph = apply_morphology(binary, MORPH_KERNEL_SIZE, use_open=USE_OPEN_OPERATION, use_close=USE_CLOSE_OPERATION)

    detected_area = calculate_defect_area(morph)
    pred_status = judge_status(detected_area, area_threshold)
    true_status = get_true_status_from_mask(mask_path)
    contours = find_defect_contours(morph)
    marked_image, defect_count = draw_defect_boxes(image, contours, MIN_CONTOUR_AREA)

    gt_area = calculate_gt_defect_area(mask_path)
    gt_ratio = calculate_gt_defect_ratio(mask_path)

    result_type = _determine_type(true_status, pred_status)

    stem = image_path.stem
    marked_path = marked_dir / f"{stem}_marked.jpg"
    marked_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(marked_path), marked_image)

    return {
        "image_name": image_path.name, "mask_name": mask_path.name,
        "true_status": true_status, "pred_status": pred_status,
        "result_type": result_type, "detected_defect_area": detected_area,
        "detected_defect_count": defect_count, "gt_defect_area": gt_area,
        "gt_defect_ratio": round(gt_ratio, 6), "marked_image_path": str(marked_path)
    }


def _determine_type(true_status, pred_status):
    if true_status == "NG" and pred_status == "NG": return "TP"
    elif true_status == "OK" and pred_status == "OK": return "TN"
    elif true_status == "OK" and pred_status == "NG": return "FP"
    elif true_status == "NG" and pred_status == "OK": return "FN"
    return "UNKNOWN"


def compute_metrics(results):
    tp = sum(1 for r in results if r["result_type"] == "TP")
    tn = sum(1 for r in results if r["result_type"] == "TN")
    fp = sum(1 for r in results if r["result_type"] == "FP")
    fn = sum(1 for r in results if r["result_type"] == "FN")
    total = len(results)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "total_count": total,
            "accuracy": (tp+tn)/total if total>0 else 0,
            "precision": tp/(tp+fp) if (tp+fp)>0 else 0,
            "recall": tp/(tp+fn) if (tp+fn)>0 else 0}


def analyze_gt_area(results):
    tp_r = [r for r in results if r["result_type"] == "TP"]
    fn_r = [r for r in results if r["result_type"] == "FN"]
    return {
        "tp_avg_gt_area": sum(r["gt_defect_area"] for r in tp_r)/len(tp_r) if tp_r else 0,
        "fn_avg_gt_area": sum(r["gt_defect_area"] for r in fn_r)/len(fn_r) if fn_r else 0,
        "tp_avg_gt_ratio": sum(r["gt_defect_ratio"] for r in tp_r)/len(tp_r) if tp_r else 0,
        "fn_avg_gt_ratio": sum(r["gt_defect_ratio"] for r in fn_r)/len(fn_r) if fn_r else 0,
    }


def main():
    print("=" * 60)
    print("完整 test 集 OpenCV 检测（基于 GT mask 标签）")
    print("=" * 60)

    summary = summarize_dataset_by_mask(TEST_DATASET_DIR)
    print(f"\n数据集概览：{summary['total_count']} 对  (OK={summary['ok_count']}, NG={summary['ng_count']})")
    print(f"GT 缺陷面积范围：{summary['defect_area_min']} ~ {summary['defect_area_max']}, 均值={summary['defect_area_mean']:.1f}")
    print(f"GT 缺陷比例均值：{summary['defect_ratio_mean']:.6f}")

    pairs = find_image_mask_pairs(TEST_DATASET_DIR)
    marked_dir = Path(MARKED_OUTPUT_DIR)
    reports_dir = Path(REAL_REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n开始检测 {len(pairs)} 对图像...")
    results = []
    for i, (img_path, mask_path) in enumerate(pairs):
        if (i + 1) % 200 == 0:
            print(f"  进度：{i + 1}/{len(pairs)}")
        r = detect_and_evaluate(img_path, mask_path, marked_dir, GRAY_THRESHOLD, THRESHOLD)
        results.append(r)

    metrics = compute_metrics(results)
    area_analysis = analyze_gt_area(results)

    tp_n = len([r for r in results if r["result_type"] == "TP"])
    tn_n = len([r for r in results if r["result_type"] == "TN"])
    fp_n = len([r for r in results if r["result_type"] == "FP"])
    fn_n = len([r for r in results if r["result_type"] == "FN"])

    print(f"\n检测完成。TP={tp_n}, TN={tn_n}, FP={fp_n}, FN={fn_n}")
    print(f"Accuracy={metrics['accuracy']*100:.1f}%, Precision={metrics['precision']*100:.1f}%, Recall={metrics['recall']*100:.1f}%")

    detail_path = reports_dir / "full_testset_detection_report.csv"
    fields = ["image_name","mask_name","true_status","pred_status","result_type",
              "detected_defect_area","detected_defect_count","gt_defect_area","gt_defect_ratio","marked_image_path"]
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(results)

    summary_path = reports_dir / "full_testset_summary_metrics.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tp","tn","fp","fn","total_count","accuracy","precision","recall"])
        w.writeheader(); w.writerow(metrics)

    area_path = reports_dir / "gt_defect_area_analysis.csv"
    with open(area_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tp_avg_gt_area","fn_avg_gt_area","tp_avg_gt_ratio","fn_avg_gt_ratio"])
        w.writeheader(); w.writerow(area_analysis)

    print(f"\nGT 缺陷面积分析：TP 平均={area_analysis['tp_avg_gt_area']:.1f}, FN 平均={area_analysis['fn_avg_gt_area']:.1f}")
    print(f"GT 缺陷比例     ：TP 平均={area_analysis['tp_avg_gt_ratio']:.6f}, FN 平均={area_analysis['fn_avg_gt_ratio']:.6f}")
    if area_analysis['tp_avg_gt_area'] > area_analysis['fn_avg_gt_area']:
        print("结论：TP 样本的平均 GT 缺陷面积 > FN 样本，漏检与缺陷面积较小有关。")
    else:
        print("结论：TP 与 FN 的 GT 缺陷面积差异不明显，漏检可能还与其他因素有关。")

    print(f"\n报表已保存：{detail_path}")
    print(f"          ：{summary_path}")
    print(f"          ：{area_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
