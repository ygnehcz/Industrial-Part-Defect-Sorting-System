"""阈值方法实验模块"""

import cv2
import numpy as np
from pathlib import Path

from config import MIN_CONTOUR_AREA
from src.image_processor import (
    read_image, convert_to_gray, threshold_image,
    calculate_defect_area, find_defect_contours, draw_defect_boxes
)
from src.morphology import apply_morphology
from src.judge import judge_status
from src.gt_mask_utils import get_true_status_from_mask


def determine_result_type(true_status, pred_status):
    if true_status == "NG" and pred_status == "NG": return "TP"
    elif true_status == "OK" and pred_status == "OK": return "TN"
    elif true_status == "OK" and pred_status == "NG": return "FP"
    elif true_status == "NG" and pred_status == "OK": return "FN"
    return "UNKNOWN"


def run_single_experiment(pairs, gray_threshold, area_threshold, kernel_size,
                          use_open=True, use_close=True, method="fixed"):
    """对给定参数组合运行完整 test 集检测，返回指标"""
    tp = tn = fp = fn = 0
    total = 0

    for img_path, mask_path in pairs:
        total += 1
        image = read_image(img_path)
        gray = convert_to_gray(image)

        if method == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "adaptive":
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        else:
            binary = threshold_image(gray, gray_threshold)

        morph = apply_morphology(binary, kernel_size, use_open=use_open, use_close=use_close)
        detected_area = calculate_defect_area(morph)
        pred_status = judge_status(detected_area, area_threshold)
        true_status = get_true_status_from_mask(mask_path)

        rt = determine_result_type(true_status, pred_status)
        if rt == "TP": tp += 1
        elif rt == "TN": tn += 1
        elif rt == "FP": fp += 1
        elif rt == "FN": fn += 1

    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {"method": method, "gray_threshold": gray_threshold, "area_threshold": area_threshold,
            "kernel_size": kernel_size, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "total_count": total, "accuracy": acc, "precision": prec, "recall": rec}


def run_all_experiments(pairs):
    """运行所有参数组合实验"""
    experiments = []

    gray_list = [60, 80, 100, 120, 160, 200]
    area_list = [20, 50, 100, 200, 500]
    kernel_list = [3, 5, 7]

    total_combos = len(gray_list) * len(area_list) * len(kernel_list) + 2
    count = 0

    # Fixed threshold grid
    for g in gray_list:
        for a in area_list:
            for k in kernel_list:
                count += 1
                print(f"  [{count}/{total_combos}] fixed: gray={g}, area={a}, kernel={k}")
                r = run_single_experiment(pairs, g, a, k, method="fixed")
                experiments.append(r)

    # Otsu
    count += 1
    print(f"  [{count}/{total_combos}] otsu")
    for a in area_list:
        r = run_single_experiment(pairs, 0, a, 5, method="otsu")
        r["gray_threshold"] = "otsu"
        experiments.append(r)

    # Adaptive
    count += 1
    print(f"  [{count}/{total_combos}] adaptive")
    for a in area_list:
        r = run_single_experiment(pairs, 0, a, 5, method="adaptive")
        r["gray_threshold"] = "adaptive"
        experiments.append(r)

    return experiments


def find_best_method(experiments):
    """找出 recall 最高的方法，recall 相同则选 precision 更高"""
    best = None
    for exp in experiments:
        if best is None:
            best = exp
        elif exp["recall"] > best["recall"]:
            best = exp
        elif exp["recall"] == best["recall"] and exp["precision"] > best["precision"]:
            best = exp
    return best
