"""真实工业缺陷数据集 OpenCV 检测模块"""

import cv2
from pathlib import Path

from config import (
    GRAY_THRESHOLD,
    THRESHOLD,
    MIN_CONTOUR_AREA,
    MORPH_KERNEL_SIZE,
    USE_OPEN_OPERATION,
    USE_CLOSE_OPERATION
)
from src.image_processor import (
    read_image,
    convert_to_gray,
    threshold_image,
    calculate_defect_area,
    find_defect_contours,
    draw_defect_boxes
)
from src.morphology import apply_morphology
from src.judge import judge_status

REAL_SAMPLES_DIR = "data/real_samples"
REAL_OUTPUT_DIR = "outputs/real_dataset"
REAL_MARKED_DIR = "outputs/real_dataset/marked"
REAL_PROCESSED_DIR = "outputs/real_dataset/processed"
REAL_REPORTS_DIR = "outputs/real_dataset/reports"


def detect_real_image(image_path, marked_dir, processed_dir, threshold):
    """对单张真实样本图片执行完整检测流程"""
    image = read_image(image_path)
    gray = convert_to_gray(image)
    binary = threshold_image(gray, GRAY_THRESHOLD)

    morph = apply_morphology(
        binary,
        MORPH_KERNEL_SIZE,
        use_open=USE_OPEN_OPERATION,
        use_close=USE_CLOSE_OPERATION
    )

    defect_area = calculate_defect_area(morph)
    pred_status = judge_status(defect_area, threshold)
    contours = find_defect_contours(morph)

    marked_image, defect_count = draw_defect_boxes(
        image, contours, MIN_CONTOUR_AREA
    )

    image_name = image_path.name
    stem_name = image_path.stem

    marked_path = marked_dir / f"{stem_name}_marked.jpg"
    processed_path = processed_dir / f"{stem_name}_morph.jpg"

    marked_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(marked_path), marked_image)
    cv2.imwrite(str(processed_path), morph)

    return {
        "image_name": image_name,
        "image_path": str(image_path),
        "true_status": None,
        "pred_status": pred_status,
        "defect_area": defect_area,
        "defect_count": defect_count,
        "result_type": None,
        "marked_image_path": str(marked_path),
        "processed_image_path": str(processed_path)
    }


def determine_result_type(true_status, pred_status):
    """根据真实标签和预测标签判断 TP / TN / FP / FN"""
    if true_status == "NG" and pred_status == "NG":
        return "TP"
    elif true_status == "OK" and pred_status == "OK":
        return "TN"
    elif true_status == "OK" and pred_status == "NG":
        return "FP"
    elif true_status == "NG" and pred_status == "OK":
        return "FN"
    else:
        return "UNKNOWN"


def run_real_dataset_detection(threshold=None):
    """对 data/real_samples 目录进行批量检测"""
    if threshold is None:
        threshold = THRESHOLD

    real_dir = Path(REAL_SAMPLES_DIR)
    marked_dir = Path(REAL_MARKED_DIR)
    processed_dir = Path(REAL_PROCESSED_DIR)

    marked_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for status_folder in ["OK", "NG"]:
        folder_path = real_dir / status_folder
        if not folder_path.exists():
            print(f"文件夹不存在：{folder_path}")
            continue

        image_files = sorted(folder_path.glob("*.png")) + sorted(folder_path.glob("*.jpg"))
        for image_path in image_files:
            result = detect_real_image(image_path, marked_dir, processed_dir, threshold)
            result["true_status"] = status_folder
            result["result_type"] = determine_result_type(status_folder, result["pred_status"])
            results.append(result)

    return results


def compute_metrics(results):
    """统计 TP / TN / FP / FN 并计算评价指标"""
    tp = sum(1 for r in results if r["result_type"] == "TP")
    tn = sum(1 for r in results if r["result_type"] == "TN")
    fp = sum(1 for r in results if r["result_type"] == "FP")
    fn = sum(1 for r in results if r["result_type"] == "FN")
    total = len(results)

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total_count": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }
