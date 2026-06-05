"""V1.2 传统视觉增强实验

运行方式（在项目根目录执行）：
    python scripts/run_enhanced_cv_experiment.py
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.gt_mask_utils import find_image_mask_pairs, get_true_status_from_mask
from src.enhanced_cv_methods import (
    apply_clahe, apply_tophat, apply_blackhat, apply_sobel, apply_canny,
    filter_contours_by_features
)
from src.image_processor import threshold_image, calculate_defect_area, find_defect_contours
from src.morphology import apply_morphology
from src.judge import judge_status

TEST_DATASET_DIR = r"D:\Industrial-Surface-Defect-Inspection\data\raw\test"
REPORTS_DIR = "outputs/real_dataset/reports"


def determine_type(true_status, pred_status):
    if true_status == "NG" and pred_status == "NG": return "TP"
    elif true_status == "OK" and pred_status == "OK": return "TN"
    elif true_status == "OK" and pred_status == "NG": return "FP"
    elif true_status == "NG" and pred_status == "OK": return "FN"
    return "UNKNOWN"


def run_method(pairs, method_name, params_str, detect_fn):
    """运行一种检测方法并返回指标"""
    tp = tn = fp = fn = 0
    for img_path, mask_path in pairs:
        pred = detect_fn(img_path)
        true = get_true_status_from_mask(mask_path)
        rt = determine_type(true, pred)
        if rt == "TP": tp += 1
        elif rt == "TN": tn += 1
        elif rt == "FP": fp += 1
        elif rt == "FN": fn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {"method": method_name, "parameters": params_str,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total_count": total,
            "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}


def main():
    print("=" * 60)
    print("V1.2 传统视觉增强实验")
    print("=" * 60)

    pairs = find_image_mask_pairs(TEST_DATASET_DIR)
    print(f"\n数据集：{len(pairs)} 对 (来自 {TEST_DATASET_DIR})")

    experiments = []
    total_exp = 0

    # ---- CLAHE + fixed threshold ----
    print("\n--- CLAHE + fixed threshold ---")
    for clip in [2.0]:
        for gray_th in [60, 100, 160]:
            for area_th in [20, 50, 100]:
                for ks in [3, 5]:
                    total_exp += 1
                    params = f"clip={clip},gray={gray_th},area={area_th},kernel={ks}"
                    print(f"  [{total_exp}] clahe_fixed: {params}")

                    def make_fn(gt=gray_th, at=area_th, k=ks, cl=clip):
                        def fn(img_path):
                            img = cv2.imread(str(img_path))
                            if img is None: return "OK"
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            enhanced = apply_clahe(gray, clip_limit=cl)
                            binary = threshold_image(enhanced, gt)
                            morph = apply_morphology(binary, k, use_open=True, use_close=True)
                            area = calculate_defect_area(morph)
                            return judge_status(area, at)
                        return fn

                    r = run_method(pairs, "clahe_fixed", params, make_fn())
                    experiments.append(r)

    # ---- CLAHE + adaptive ----
    print("\n--- CLAHE + adaptive threshold ---")
    for clip in [2.0]:
        for area_th in [20, 50, 100]:
            for ks in [5]:
                total_exp += 1
                params = f"clip={clip},area={area_th},kernel={ks}"
                print(f"  [{total_exp}] clahe_adaptive: {params}")

                def make_fn(at=area_th, k=ks, cl=clip):
                    def fn(img_path):
                        img = cv2.imread(str(img_path))
                        if img is None: return "OK"
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        enhanced = apply_clahe(gray, clip_limit=cl)
                        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY, 11, 2)
                        morph = apply_morphology(binary, k, use_open=True, use_close=True)
                        area = calculate_defect_area(morph)
                        return judge_status(area, at)
                    return fn

                r = run_method(pairs, "clahe_adaptive", params, make_fn())
                experiments.append(r)

    # ---- Top-hat + fixed threshold ----
    print("\n--- Top-hat + fixed threshold ---")
    for ksz in [9, 15]:
        for gray_th in [60, 100]:
            for area_th in [20, 50]:
                for mk in [5]:
                    total_exp += 1
                    params = f"tophat_k={ksz},gray={gray_th},area={area_th},kernel={mk}"
                    print(f"  [{total_exp}] tophat_fixed: {params}")

                    def make_fn(gt=gray_th, at=area_th, tks=ksz, k=mk):
                        def fn(img_path):
                            img = cv2.imread(str(img_path))
                            if img is None: return "OK"
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            enhanced = apply_tophat(gray, kernel_size=tks)
                            binary = threshold_image(enhanced, gt)
                            morph = apply_morphology(binary, k, use_open=True, use_close=True)
                            area = calculate_defect_area(morph)
                            return judge_status(area, at)
                        return fn

                    r = run_method(pairs, "tophat_fixed", params, make_fn())
                    experiments.append(r)

    # ---- Black-hat + fixed threshold ----
    print("\n--- Black-hat + fixed threshold ---")
    for ksz in [9, 15]:
        for gray_th in [60, 100]:
            for area_th in [20, 50]:
                for mk in [5]:
                    total_exp += 1
                    params = f"blackhat_k={ksz},gray={gray_th},area={area_th},kernel={mk}"
                    print(f"  [{total_exp}] blackhat_fixed: {params}")

                    def make_fn(gt=gray_th, at=area_th, bks=ksz, k=mk):
                        def fn(img_path):
                            img = cv2.imread(str(img_path))
                            if img is None: return "OK"
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            enhanced = apply_blackhat(gray, kernel_size=bks)
                            binary = threshold_image(enhanced, gt)
                            morph = apply_morphology(binary, k, use_open=True, use_close=True)
                            area = calculate_defect_area(morph)
                            return judge_status(area, at)
                        return fn

                    r = run_method(pairs, "blackhat_fixed", params, make_fn())
                    experiments.append(r)

    # ---- Sobel + fixed threshold ----
    print("\n--- Sobel + fixed threshold ---")
    for gray_th in [30, 60, 100]:
        for area_th in [20, 50, 100]:
            for mk in [5]:
                total_exp += 1
                params = f"gray={gray_th},area={area_th},kernel={mk}"
                print(f"  [{total_exp}] sobel_fixed: {params}")

                def make_fn(gt=gray_th, at=area_th, k=mk):
                    def fn(img_path):
                        img = cv2.imread(str(img_path))
                        if img is None: return "OK"
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        edges = apply_sobel(gray)
                        binary = threshold_image(edges, gt)
                        morph = apply_morphology(binary, k, use_open=True, use_close=True)
                        area = calculate_defect_area(morph)
                        return judge_status(area, at)
                    return fn

                r = run_method(pairs, "sobel_fixed", params, make_fn())
                experiments.append(r)

    # ---- Canny + connected component filtering ----
    print("\n--- Canny + component filtering ---")
    for low_th in [30, 50]:
        for high_th in [100, 150]:
            for min_area in [20, 50]:
                total_exp += 1
                params = f"canny_low={low_th},canny_high={high_th},min_area={min_area}"
                print(f"  [{total_exp}] canny_filter: {params}")

                def make_fn(lt=low_th, ht=high_th, ma=min_area):
                    def fn(img_path):
                        img = cv2.imread(str(img_path))
                        if img is None: return "OK"
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        edges = apply_canny(gray, low_threshold=lt, high_threshold=ht)
                        contours = find_defect_contours(edges)
                        filtered = filter_contours_by_features(contours, min_area=ma)
                        area = sum(cv2.contourArea(c) for c in filtered)
                        return judge_status(area, 1)
                    return fn

                r = run_method(pairs, "canny_filter", params, make_fn())
                experiments.append(r)

    # ---- Save results ----
    reports_dir = Path(REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "enhanced_cv_experiment.csv"
    fields = ["method", "parameters", "tp", "tn", "fp", "fn", "total_count",
              "accuracy", "precision", "recall", "f1_score"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(experiments)

    # ---- Analysis ----
    best_f1 = max(experiments, key=lambda x: x["f1_score"])

    # Recall >= 50% with highest precision
    high_recall = [e for e in experiments if e["recall"] >= 0.5]
    best_high_recall = max(high_recall, key=lambda x: x["precision"]) if high_recall else None

    print(f"\n{'='*60}")
    print(f"实验完成，共 {len(experiments)} 组。报表：{csv_path}")
    print(f"\n最佳 F1 方法：{best_f1['method']}, {best_f1['parameters']}")
    print(f"  TP={best_f1['tp']}, TN={best_f1['tn']}, FP={best_f1['fp']}, FN={best_f1['fn']}")
    print(f"  Acc={best_f1['accuracy']*100:.1f}%, Prec={best_f1['precision']*100:.1f}%, Rec={best_f1['recall']*100:.1f}%, F1={best_f1['f1_score']:.4f}")

    if best_high_recall:
        print(f"\nRecall >= 50% 且 Precision 最高：{best_high_recall['method']}, {best_high_recall['parameters']}")
        print(f"  TP={best_high_recall['tp']}, TN={best_high_recall['tn']}, FP={best_high_recall['fp']}, FN={best_high_recall['fn']}")
        print(f"  Acc={best_high_recall['accuracy']*100:.1f}%, Prec={best_high_recall['precision']*100:.1f}%, Rec={best_high_recall['recall']*100:.1f}%, F1={best_high_recall['f1_score']:.4f}")
    else:
        print(f"\nRecall >= 50%：无方法达到")

    # Top 5 by F1
    print(f"\nTop 5 (按 F1 降序)：")
    top5 = sorted(experiments, key=lambda x: -x["f1_score"])[:5]
    for i, e in enumerate(top5):
        print(f"  {i+1}. {e['method']:16s} {e['parameters']:35s}  "
              f"TP={e['tp']:3d} TN={e['tn']:3d} FP={e['fp']:3d} FN={e['fn']:3d}  "
              f"F1={e['f1_score']:.4f}  Rec={e['recall']*100:.1f}%")

    # ---- Comparison with V1.1 baseline ----
    v1_1_baseline = {"recall": 5/110, "precision": 5/5 if 5>0 else 0, "f1_score": 2*(5/5)*(5/110)/((5/5)+(5/110)) if (5/5)+(5/110)>0 else 0}
    print(f"\nV1.1 基线: Rec={v1_1_baseline['recall']*100:.1f}%, Prec={v1_1_baseline['precision']*100:.1f}%, F1={v1_1_baseline['f1_score']:.4f}")
    v1_1_best_recall = {"recall": 1.0, "precision": 110/(110+853), "f1_score": 2*1.0*(110/(110+853))/(1.0+(110/(110+853)))}
    print(f"V1.1 最佳 Recall: Rec=100.0%, Prec=11.4%, F1={v1_1_best_recall['f1_score']:.4f}")
    print(f"\nV1.2 最佳 F1:  Rec={best_f1['recall']*100:.1f}%, Prec={best_f1['precision']*100:.1f}%, F1={best_f1['f1_score']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
