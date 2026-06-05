import csv

import matplotlib.pyplot as plt


def load_threshold_experiment(csv_path):
    experiment_results = []

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            result = {
                "threshold": int(row["threshold"]),
                "accuracy": float(row["accuracy"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"])
            }

            experiment_results.append(result)

    return experiment_results


def plot_threshold_metrics(experiment_results, output_path):
    thresholds = []
    accuracies = []
    precisions = []
    recalls = []

    for result in experiment_results:
        thresholds.append(result["threshold"])
        accuracies.append(result["accuracy"] * 100)
        precisions.append(result["precision"] * 100)
        recalls.append(result["recall"] * 100)

    plt.figure(figsize=(8, 5))

    plt.plot(thresholds, accuracies, marker="o", label="Accuracy")
    plt.plot(thresholds, precisions, marker="o", label="Precision")
    plt.plot(thresholds, recalls, marker="o", label="Recall")

    plt.xlabel("Threshold")
    plt.ylabel("Score (%)")
    plt.title("Threshold Experiment Metrics")
    plt.legend()
    plt.grid(True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()