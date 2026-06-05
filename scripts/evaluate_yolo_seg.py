"""YOLO Segmentation 图像级评价脚本

对 val 集进行推理，统计图像级 TP/TN/FP/FN。
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

YOLO_DIR = Path("data/yolo_seg")
VAL_IMG_DIR = YOLO_DIR / "images" / "val"
VAL_LBL_DIR = YOLO_DIR / "labels" / "val"
BEST_PT = Path("outputs/yolo_seg/runs/yolov8n_seg_smoke/weights/best.pt")
REPORT_DIR = Path("outputs/yolo_seg/reports")
VIZ_DIR = Path("docs/images/yolo_seg_predictions")


def get_true_status(label_path):
    content = label_path.read_text().strip() if label_path.exists() else ""
    return "NG" if content else "OK"


def main():
    print("=" * 60)
    print("YOLO Segmentation 图像级评价")
    print("=" * 60)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    if not BEST_PT.exists():
        print(f"\n错误：模型不存在 {BEST_PT}")
        print("请先运行训练。")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics 未安装")
        return

    print(f"\n模型：{BEST_PT}")
    print(f"val 图片目录：{VAL_IMG_DIR}")

    model = YOLO(str(BEST_PT))

    tp = tn = fp = fn = 0
    results = []

    img_files = sorted(VAL_IMG_DIR.glob("*.png"))
    print(f"\n推理 {len(img_files)} 张 val 图片...")

    for i, img_path in enumerate(img_files):
        stem = img_path.stem
        lbl_path = VAL_LBL_DIR / f"{stem}.txt"
        true_status = get_true_status(lbl_path)

        pred_result = model(str(img_path), verbose=False)
        if pred_result and pred_result[0].masks is not None:
            pred_status = "NG"
        else:
            pred_status = "OK"

        rt = "UNKNOWN"
        if true_status == "NG" and pred_status == "NG":
            rt = "TP"; tp += 1
        elif true_status == "OK" and pred_status == "OK":
            rt = "TN"; tn += 1
        elif true_status == "OK" and pred_status == "NG":
            rt = "FP"; fp += 1
        elif true_status == "NG" and pred_status == "OK":
            rt = "FN"; fn += 1

        results.append({
            "image_name": img_path.name, "true_status": true_status,
            "pred_status": pred_status, "result_type": rt
        })

        if (i+1) % 200 == 0:
            print(f"  进度：{i+1}/{len(img_files)}")

    total = tp + tn + fp + fn
    acc = (tp+tn)/total if total>0 else 0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0

    print(f"\n图像级评价结果：")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Accuracy={acc*100:.1f}%, Precision={prec*100:.1f}%, Recall={rec*100:.1f}%, F1={f1:.4f}")

    # Save reports
    detail_path = REPORT_DIR / "yolo_seg_eval_report.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_name","true_status","pred_status","result_type"])
        w.writeheader(); w.writerows(results)

    summary_path = REPORT_DIR / "yolo_seg_summary_metrics.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tp","tn","fp","fn","total_count","accuracy","precision","recall","f1_score"])
        w.writeheader(); w.writerow({"tp":tp,"tn":tn,"fp":fp,"fn":fn,"total_count":total,
                                      "accuracy":acc,"precision":prec,"recall":rec,"f1_score":f1})

    # Save prediction visualizations
    tp_samples = [r for r in results if r["result_type"]=="TP"][:2]
    fn_samples = [r for r in results if r["result_type"]=="FN"][:2]
    fp_samples = [r for r in results if r["result_type"]=="FP"][:2]

    for label, samples in [("TP",tp_samples),("FN",fn_samples),("FP",fp_samples)]:
        for j, sample in enumerate(samples):
            img_path = VAL_IMG_DIR / sample["image_name"]
            pred = model(str(img_path), verbose=False)
            out_path = VIZ_DIR / f"{label.lower()}_case_{j+1:02d}.png"
            if pred and pred[0].masks is not None:
                annotated = pred[0].plot()
                cv2.imwrite(str(out_path), annotated)
                print(f"  可视化 {label}: {out_path.name}")

    print(f"\n报表：{detail_path}")
    print(f"     ：{summary_path}")
    print(f"可视化：{VIZ_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
