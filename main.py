from pathlib import Path

from config import (
    SAMPLE_IMAGE_DIR,
    IMAGE_OUTPUT_DIR,
    REPORT_DIR,
    THRESHOLD,
    THRESHOLD_LIST,
    THRESHOLD_EXPERIMENT_CSV,
    THRESHOLD_METRICS_FIGURE,
    MORPH_OUTPUT_DIR
)
from src.batch_detector import (
    run_batch_detection,
    run_threshold_experiments,
    find_best_threshold
)
from src.evaluator import evaluate_results
from src.report import save_report, save_metrics, save_threshold_experiments
from src.visualizer import load_threshold_experiment, plot_threshold_metrics


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


def main():
    sample_dir = Path(SAMPLE_IMAGE_DIR)
    image_output_dir = Path(IMAGE_OUTPUT_DIR)
    report_dir = Path(REPORT_DIR)
    morph_output_dir = Path(MORPH_OUTPUT_DIR)

    image_output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    morph_output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(sample_dir.glob("*.jpg"))

    print("====== 当前阈值批量检测 ======")

    results = run_batch_detection(image_paths, image_output_dir, morph_output_dir, THRESHOLD)

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

    experiment_results = run_threshold_experiments(image_paths, image_output_dir, morph_output_dir)

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
    print(f"形态学结果图保存目录：{morph_output_dir}")


if __name__ == "__main__":
    main()
