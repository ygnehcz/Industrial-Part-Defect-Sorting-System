from pathlib import Path

import cv2

from config import (
    GRAY_THRESHOLD,
    THRESHOLD_LIST,
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
from src.evaluator import evaluate_results


def get_true_status_from_filename(image_path):
    image_name = image_path.name.upper()

    if "_NG" in image_name:
        return "NG"
    elif "_OK" in image_name:
        return "OK"
    else:
        return "UNKNOWN"


def detect_single_image(image_path, image_output_dir, morph_output_dir, threshold):
    image = read_image(image_path)

    gray_image = convert_to_gray(image)

    binary_image = threshold_image(gray_image, GRAY_THRESHOLD)

    morph_image = apply_morphology(
        binary_image,
        MORPH_KERNEL_SIZE,
        use_open=USE_OPEN_OPERATION,
        use_close=USE_CLOSE_OPERATION
    )

    defect_area = calculate_defect_area(morph_image)

    pred_status = judge_status(defect_area, threshold)

    true_status = get_true_status_from_filename(image_path)

    contours = find_defect_contours(morph_image)

    marked_image, defect_count = draw_defect_boxes(
        image,
        contours,
        MIN_CONTOUR_AREA
    )

    image_name = image_path.name
    stem_name = image_path.stem

    binary_output_path = image_output_dir / f"{stem_name}_binary.jpg"
    marked_output_path = image_output_dir / f"{stem_name}_marked.jpg"
    morph_output_path = morph_output_dir / f"{stem_name}_morph.jpg"

    cv2.imwrite(str(binary_output_path), binary_image)
    cv2.imwrite(str(marked_output_path), marked_image)
    cv2.imwrite(str(morph_output_path), morph_image)

    result = {
        "image_name": image_name,
        "defect_area": defect_area,
        "defect_count": defect_count,
        "true_status": true_status,
        "pred_status": pred_status,
        "marked_image_path": str(marked_output_path),
        "binary_image_path": str(binary_output_path),
        "morph_image_path": str(morph_output_path)
    }

    return result


def run_batch_detection(image_paths, image_output_dir, morph_output_dir, threshold):
    results = []

    for image_path in image_paths:
        result = detect_single_image(image_path, image_output_dir, morph_output_dir, threshold)
        results.append(result)

    return results


def run_threshold_experiments(image_paths, image_output_dir, morph_output_dir):
    experiment_results = []

    for threshold in THRESHOLD_LIST:
        results = run_batch_detection(image_paths, image_output_dir, morph_output_dir, threshold)
        metrics = evaluate_results(results)

        experiment_result = {
            "threshold": threshold,
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "total_count": metrics["total_count"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"]
        }

        experiment_results.append(experiment_result)

    return experiment_results


def find_best_threshold(experiment_results):
    best_result = None

    for result in experiment_results:
        if best_result is None:
            best_result = result
        elif result["recall"] > best_result["recall"]:
            best_result = result
        elif result["recall"] == best_result["recall"] and result["precision"] > best_result["precision"]:
            best_result = result

    return best_result
