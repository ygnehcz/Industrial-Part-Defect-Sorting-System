"""YOLO Segmentation confidence 阈值实验"""

import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BEST_PT = Path("outputs/yolo_seg/runs/yolov8n_seg_smoke/weights/best.pt")
VAL_IMG_DIR = Path("data/yolo_seg/images/val")
VAL_LBL_DIR = Path("data/yolo_seg/labels/val")
REPORT_DIR = Path("outputs/yolo_seg/reports")

def get_true_status(label_path):
    content = label_path.read_text().strip() if label_path.exists() else ""
    return "NG" if content else "OK"

def main():
    print("=" * 60)
    print("YOLO Confidence 阈值实验")
    print("=" * 60)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(str(BEST_PT))

    conf_list = [0.05, 0.1, 0.25, 0.5, 0.75]
    img_files = sorted(VAL_IMG_DIR.glob("*.png"))
    experiments = []

    for conf in conf_list:
        tp = tn = fp = fn = 0
        for img_path in img_files:
            stem = img_path.stem
            lbl_path = VAL_LBL_DIR / f"{stem}.txt"
            true_status = get_true_status(lbl_path)
            pred_result = model(str(img_path), conf=conf, verbose=False)
            has_mask = pred_result and pred_result[0].masks is not None
            pred_status = "NG" if has_mask else "OK"

            if true_status=="NG" and pred_status=="NG": tp+=1
            elif true_status=="OK" and pred_status=="OK": tn+=1
            elif true_status=="OK" and pred_status=="NG": fp+=1
            elif true_status=="NG" and pred_status=="OK": fn+=1

        total = tp+tn+fp+fn
        acc = (tp+tn)/total if total>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0

        experiments.append({"conf": conf, "tp":tp,"tn":tn,"fp":fp,"fn":fn,
                            "total":total,"accuracy":acc,"precision":prec,"recall":rec,"f1_score":f1})
        print(f"  conf={conf:.2f}: TP={tp}, TN={tn}, FP={fp}, FN={fn}, F1={f1:.4f}")

    # Save
    csv_path = REPORT_DIR / "yolo_confidence_experiment.csv"
    fields = ["conf","tp","tn","fp","fn","total","accuracy","precision","recall","f1_score"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(experiments)

    best_f1 = max(experiments, key=lambda x: x["f1_score"])
    best_rec = max(experiments, key=lambda x: x["recall"])
    print(f"\n最佳 F1：conf={best_f1['conf']:.2f}, F1={best_f1['f1_score']:.4f}, Prec={best_f1['precision']*100:.1f}%, Rec={best_f1['recall']*100:.1f}%")
    print(f"最佳 Recall：conf={best_rec['conf']:.2f}, Rec={best_rec['recall']*100:.1f}%, Prec={best_rec['precision']*100:.1f}%")
    print(f"报表：{csv_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
