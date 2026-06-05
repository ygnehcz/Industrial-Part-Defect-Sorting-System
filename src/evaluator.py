def evaluate_results(results):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for result in results:
        true_status = result["true_status"]
        pred_status = result["pred_status"]

        if true_status == "NG" and pred_status == "NG":
            tp = tp + 1
        elif true_status == "OK" and pred_status == "OK":
            tn = tn + 1
        elif true_status == "OK" and pred_status == "NG":
            fp = fp + 1
        elif true_status == "NG" and pred_status == "OK":
            fn = fn + 1

    total_count = tp + tn + fp + fn

    if total_count > 0:
        accuracy = (tp + tn) / total_count
    else:
        accuracy = 0

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    metrics = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total_count": total_count,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

    return metrics