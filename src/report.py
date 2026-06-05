import csv


def save_report(results, output_path):
    fieldnames = [
        "image_name",
        "defect_area",
        "defect_count",
        "true_status",
        "pred_status",
        "marked_image_path",
        "binary_image_path",
        "morph_image_path"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def save_metrics(metrics, output_path):
    fieldnames = [
        "tp",
        "tn",
        "fp",
        "fn",
        "total_count",
        "accuracy",
        "precision",
        "recall"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(metrics)


def save_threshold_experiments(experiment_results, output_path):
    fieldnames = [
        "threshold",
        "tp",
        "tn",
        "fp",
        "fn",
        "total_count",
        "accuracy",
        "precision",
        "recall"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(experiment_results)
