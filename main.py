from pathlib import Path

import cv2

from config import (
    SAMPLE_IMAGE_DIR,
    IMAGE_OUTPUT_DIR,
    REPORT_DIR,
    GRAY_THRESHOLD,
    THRESHOLD,
    THRESHOLD_LIST,
    MIN_CONTOUR_AREA,
    THRESHOLD_EXPERIMENT_CSV,
    THRESHOLD_METRICS_FIGURE
)
from src.image_processor import (
    read_image,
    convert_to_gray,
    threshold_image,
    calculate_defect_area,
    find_defect_contours,
    draw_defect_boxes
)
from src.judge import judge_status
from src.evaluator import evaluate_results
from src.report import save_report, save_metrics, save_threshold_experiments
from src.visualizer import load_threshold_experiment, plot_threshold_metrics


def get_true_status_from_filename(image_path):
    image_name = image_path.name.upper()

    if "_NG" in image_name:
        return "NG"
    elif "_OK" in image_name:
        return "OK"
    else:
        return "UNKNOWN"


def detect_single_image(image_path, image_output_dir, threshold):
    image = read_image(image_path)

    gray_image = convert_to_gray(image)

    binary_image = threshold_image(gray_image, GRAY_THRESHOLD)

    defect_area = calculate_defect_area(binary_image)

    pred_status = judge_status(defect_area, threshold)

    true_status = get_true_status_from_filename(image_path)

    contours = find_defect_contours(binary_image)

    marked_image, defect_count = draw_defect_boxes(
        image,
        contours,
        MIN_CONTOUR_AREA
    )

    image_name = image_path.name
    stem_name = image_path.stem

    binary_output_path = image_output_dir / f"{stem_name}_binary.jpg"
    marked_output_path = image_output_dir / f"{stem_name}_marked.jpg"

    cv2.imwrite(str(binary_output_path), binary_image)
    cv2.imwrite(str(marked_output_path), marked_image)

    result = {
        "image_name": image_name,
        "defect_area": defect_area,
        "defect_count": defect_count,
        "true_status": true_status,
        "pred_status": pred_status,
        "marked_image_path": str(marked_output_path),
        "binary_image_path": str(binary_output_path)
    }

    return result


def run_batch_detection(image_paths, image_output_dir, threshold):
    results = []

    for image_path in image_paths:
        result = detect_single_image(image_path, image_output_dir, threshold)
        results.append(result)

    return results


def print_single_result(result):
    print(
        f"图片：{result['image_name']}，"
        f"真实：{result['true_status']}，"
        f"预测：{result['pred_status']}，"
        f"缺陷面积：{result['defect_area']}，"
        f"缺陷数量：{result['defect_count']}"
    )


def print_metrics(metrics):
    print(f"TP：{metrics['tp']}")
    print(f"TN：{metrics['tn']}")
    print(f"FP：{metrics['fp']}")
    print(f"FN：{metrics['fn']}")
    print(f"总数量：{metrics['total_count']}")
    print(f"Accuracy准确率：{metrics['accuracy'] * 100:.1f}%")
    print(f"Precision精确率：{metrics['precision'] * 100:.1f}%")
    print(f"Recall召回率：{metrics['recall'] * 100:.1f}%")


def run_threshold_experiments(image_paths, image_output_dir):
    experiment_results = []

    for threshold in THRESHOLD_LIST:
        results = run_batch_detection(image_paths, image_output_dir, threshold)
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

        print(f"====== 阈值实验：{threshold} ======")
        print_metrics(metrics)

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


def main():
    sample_dir = Path(SAMPLE_IMAGE_DIR)
    image_output_dir = Path(IMAGE_OUTPUT_DIR)
    report_dir = Path(REPORT_DIR)

    image_output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(sample_dir.glob("*.jpg"))

    print("====== 当前阈值批量检测 ======")

    results = run_batch_detection(image_paths, image_output_dir, THRESHOLD)

    for result in results:
        print_single_result(result)

    metrics = evaluate_results(results)

    detail_report_path = report_dir / "batch_detection_report.csv"
    summary_report_path = report_dir / "batch_summary_metrics.csv"

    save_report(results, detail_report_path)
    save_metrics(metrics, summary_report_path)

    print("====== 当前阈值评价结果 ======")
    print(f"当前阈值：{THRESHOLD}")
    print_metrics(metrics)

    print("====== 多阈值参数实验 ======")

    experiment_results = run_threshold_experiments(image_paths, image_output_dir)

    threshold_report_path = Path(THRESHOLD_EXPERIMENT_CSV)
    save_threshold_experiments(experiment_results, threshold_report_path)

    loaded_experiments = load_threshold_experiment(threshold_report_path)

    figure_path = Path(THRESHOLD_METRICS_FIGURE)
    plot_threshold_metrics(loaded_experiments, figure_path)

    best_result = find_best_threshold(experiment_results)

    print("====== 推荐阈值 ======")
    print(f"推荐阈值：{best_result['threshold']}")
    print(f"Accuracy准确率：{best_result['accuracy'] * 100:.1f}%")
    print(f"Precision精确率：{best_result['precision'] * 100:.1f}%")
    print(f"Recall召回率：{best_result['recall'] * 100:.1f}%")

    print("====== 报表保存完成 ======")
    print(f"检测明细报表：{detail_report_path}")
    print(f"汇总评价报表：{summary_report_path}")
    print(f"阈值实验报表：{threshold_report_path}")
    print(f"阈值实验曲线图：{figure_path}")
    print(f"标注图保存目录：{image_output_dir}")


if __name__ == "__main__":
    main()