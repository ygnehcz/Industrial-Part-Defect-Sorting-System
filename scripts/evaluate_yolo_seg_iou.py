"""YOLO Segmentation IoU / Dice 评价脚本

对 val 集推理并计算预测 mask 与 GT mask 的像素级重合指标。
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

BEST_PT = Path("outputs/yolo_seg/runs/yolov8n_seg_smoke/weights/best.pt")
VAL_IMG_DIR = Path("data/yolo_seg/images/val")
GT_MASK_DIR = Path(r"D:\Industrial-Surface-Defect-Inspection\data\raw\test")
REPORT_DIR = Path("outputs/yolo_seg/reports")


def get_true_status_from_gt(mask_path):
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt is None: return "UNKNOWN", 0
    area = int(np.count_nonzero(gt))
    return ("NG" if area > 0 else "OK"), area


def compute_iou_dice(gt_mask, pred_mask):
    gt_bin = (gt_mask > 0).astype(np.uint8)
    pred_bin = (pred_mask > 0).astype(np.uint8)
    inter = np.sum(gt_bin & pred_bin)
    union = np.sum(gt_bin | pred_bin)
    iou = inter / union if union > 0 else 0.0
    dice = 2 * inter / (np.sum(gt_bin) + np.sum(pred_bin)) if (np.sum(gt_bin) + np.sum(pred_bin)) > 0 else 0.0
    return iou, dice, int(inter), int(union)


def main():
    print("=" * 60)
    print("YOLO Segmentation IoU / Dice 评价")
    print("=" * 60)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(str(BEST_PT))
    print(f"模型：{BEST_PT}")

    img_files = sorted(VAL_IMG_DIR.glob("*.png"))
    results = []

    for i, img_path in enumerate(img_files):
        stem = img_path.stem
        gt_path = GT_MASK_DIR / f"{stem}_GT.png"
        true_status, gt_area = get_true_status_from_gt(gt_path)

        pred_result = model(str(img_path), verbose=False)
        pred_status = "OK"
        pred_area = 0
        conf_max = 0.0
        mask_count = 0

        if pred_result and pred_result[0].masks is not None:
            masks = pred_result[0].masks.data.cpu().numpy()
            confs = pred_result[0].boxes.conf.cpu().numpy() if pred_result[0].boxes is not None else []
            mask_count = len(masks)
            if mask_count > 0:
                pred_status = "NG"
                conf_max = float(max(confs)) if len(confs) > 0 else 0.0
                h, w = pred_result[0].orig_shape
                pred_mask = np.zeros((h, w), dtype=np.uint8)
                for m in masks:
                    resized = cv2.resize(m.astype(np.uint8), (w, h))
                    pred_mask = np.maximum(pred_mask, resized)
                pred_area = int(np.count_nonzero(pred_mask))

                gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                if gt is not None:
                    gt = cv2.resize(gt, (w, h))
                    iou, dice, inter, union = compute_iou_dice(gt, pred_mask)
                else:
                    iou, dice, inter, union = 0.0, 0.0, 0, 0
            else:
                iou, dice, inter, union = 0.0, 0.0, 0, 0
        else:
            iou, dice, inter, union = 0.0, 0.0, 0, 0

        rt = "UNKNOWN"
        if true_status == "NG" and pred_status == "NG": rt = "TP"
        elif true_status == "OK" and pred_status == "OK": rt = "TN"
        elif true_status == "OK" and pred_status == "NG": rt = "FP"
        elif true_status == "NG" and pred_status == "OK": rt = "FN"

        results.append({
            "image_name": img_path.name, "true_status": true_status, "pred_status": pred_status,
            "result_type": rt, "gt_area": gt_area, "pred_area": pred_area,
            "intersection_area": inter, "union_area": union,
            "iou": round(iou, 6), "dice": round(dice, 6),
            "confidence_max": round(conf_max, 6), "pred_mask_count": mask_count
        })

        if (i+1) % 200 == 0:
            print(f"  进度：{i+1}/{len(img_files)}")

    # Image-level metrics
    tp = sum(1 for r in results if r["result_type"]=="TP")
    tn = sum(1 for r in results if r["result_type"]=="TN")
    fp = sum(1 for r in results if r["result_type"]=="FP")
    fn = sum(1 for r in results if r["result_type"]=="FN")
    total = len(results)
    acc = (tp+tn)/total if total>0 else 0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0

    ng_results = [r for r in results if r["true_status"]=="NG"]
    tp_results = [r for r in results if r["result_type"]=="TP"]

    mean_iou_ng = sum(r["iou"] for r in ng_results)/len(ng_results) if ng_results else 0
    mean_dice_ng = sum(r["dice"] for r in ng_results)/len(ng_results) if ng_results else 0
    mean_iou_tp = sum(r["iou"] for r in tp_results)/len(tp_results) if tp_results else 0
    mean_dice_tp = sum(r["dice"] for r in tp_results)/len(tp_results) if tp_results else 0
    mean_pred_area_tp = sum(r["pred_area"] for r in tp_results)/len(tp_results) if tp_results else 0
    mean_gt_area_tp = sum(r["gt_area"] for r in tp_results)/len(tp_results) if tp_results else 0
    fn_results = [r for r in results if r["result_type"]=="FN"]
    mean_gt_area_fn = sum(r["gt_area"] for r in fn_results)/len(fn_results) if fn_results else 0

    print(f"\n图像级：TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Accuracy={acc*100:.1f}%, Precision={prec*100:.1f}%, Recall={rec*100:.1f}%, F1={f1:.4f}")
    print(f"\n分割质量（NG 图）：mean IoU={mean_iou_ng:.4f}, mean Dice={mean_dice_ng:.4f}")
    print(f"分割质量（TP 图）：mean IoU={mean_iou_tp:.4f}, mean Dice={mean_dice_tp:.4f}")
    print(f"TP mean pred_area={mean_pred_area_tp:.0f}, TP mean gt_area={mean_gt_area_tp:.0f}")
    print(f"FN mean gt_area={mean_gt_area_fn:.0f}")

    # Save detail CSV
    detail_path = REPORT_DIR / "yolo_seg_iou_eval_report.csv"
    fields = ["image_name","true_status","pred_status","result_type","gt_area","pred_area",
              "intersection_area","union_area","iou","dice","confidence_max","pred_mask_count"]
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(results)

    # Save summary CSV
    summary_path = REPORT_DIR / "yolo_seg_iou_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tp","tn","fp","fn","total_count","accuracy","precision","recall","f1_score",
                                          "mean_iou_ng","mean_dice_ng","mean_iou_tp","mean_dice_tp",
                                          "mean_pred_area_tp","mean_gt_area_tp","mean_gt_area_fn"])
        w.writeheader(); w.writerow({"tp":tp,"tn":tn,"fp":fp,"fn":fn,"total_count":total,
                                      "accuracy":acc,"precision":prec,"recall":rec,"f1_score":f1,
                                      "mean_iou_ng":mean_iou_ng,"mean_dice_ng":mean_dice_ng,
                                      "mean_iou_tp":mean_iou_tp,"mean_dice_tp":mean_dice_tp,
                                      "mean_pred_area_tp":mean_pred_area_tp,"mean_gt_area_tp":mean_gt_area_tp,
                                      "mean_gt_area_fn":mean_gt_area_fn})

    print(f"\n报表：{detail_path}")
    print(f"     ：{summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
