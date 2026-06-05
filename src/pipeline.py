from pathlib import Path

from config import THRESHOLD, REPORT_DIR
from src.judge import judge_status
from src.evaluator import evaluate_results
from src.report import save_report, save_metrics


def generate_results(raw_data):
    results = []

    for item in raw_data:
        image_name = item["image_name"]
        defect_area = item["defect_area"]
        true_status = item["true_status"]

        pred_status = judge_status(defect_area, THRESHOLD)

        result = {
            "image_name": image_name,
            "defect_area": defect_area,
            "true_status": true_status,
            "pred_status": pred_status
        }

        results.append(result)

    return results


def print_metrics(metrics, report_path, summary_path):
    print("====== 检测评价结果 ======")
    print(f"TP：{metrics['tp']}")
    print(f"TN：{metrics['tn']}")
    print(f"FP：{metrics['fp']}")
    print(f"FN：{metrics['fn']}")
    print(f"总数量：{metrics['total_count']}")
    print(f"Accuracy准确率：{metrics['accuracy'] * 100:.1f}%")
    print(f"Precision精确率：{metrics['precision'] * 100:.1f}%")
    print(f"Recall召回率：{metrics['recall'] * 100:.1f}%")
    print(f"检测明细已保存：{report_path}")
    print(f"汇总指标已保存：{summary_path}")


def run_pipeline(raw_data):
    results = generate_results(raw_data)

    metrics = evaluate_results(results)

    report_dir = Path(REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "evaluation_report.csv"
    summary_path = report_dir / "summary_metrics.csv"

    save_report(results, report_path)
    save_metrics(metrics, summary_path)

    print_metrics(metrics, report_path, summary_path)