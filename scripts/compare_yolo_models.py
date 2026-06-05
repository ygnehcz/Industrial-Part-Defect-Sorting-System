"""YOLO 模型对比脚本 — 比较 V2.2 (3ep) vs V2.3 (10ep)"""

import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VAL_IMG_DIR = Path("data/yolo_seg/images/val")
VAL_LBL_DIR = Path("data/yolo_seg/labels/val")
REPORT_DIR = Path("outputs/yolo_seg/reports")

MODELS = [
    ("V2.2_3ep", "outputs/yolo_seg/runs/yolov8n_seg_smoke/weights/best.pt"),
    ("V2.3_10ep", "outputs/yolo_seg/runs/yolov8n_seg_10epochs/weights/best.pt"),
]
CONF_LIST = [0.01, 0.03, 0.05, 0.1, 0.25, 0.5]

def get_true(label_path):
    return "NG" if (label_path.exists() and label_path.read_text().strip()) else "OK"

def main():
    print("="*60)
    print("YOLO 模型对比")
    print("="*60)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    img_files = sorted(VAL_IMG_DIR.glob("*.png"))
    rows = []

    for model_name, model_path in MODELS:
        model = YOLO(str(model_path))
        for conf in CONF_LIST:
            tp=tn=fp=fn=0
            for img_path in img_files:
                stem = img_path.stem
                true = get_true(VAL_LBL_DIR / f"{stem}.txt")
                pred_result = model(str(img_path), conf=conf, verbose=False)
                has_mask = pred_result and pred_result[0].masks is not None
                pred = "NG" if has_mask else "OK"

                if true=="NG" and pred=="NG": tp+=1
                elif true=="OK" and pred=="OK": tn+=1
                elif true=="OK" and pred=="NG": fp+=1
                elif true=="NG" and pred=="OK": fn+=1

            total=tp+tn+fp+fn
            acc=(tp+tn)/total; prec=tp/(tp+fp) if(tp+fp)>0 else 0
            rec=tp/(tp+fn) if(tp+fn)>0 else 0; f1=2*prec*rec/(prec+rec) if(prec+rec)>0 else 0
            rows.append({"model":model_name,"model_path":model_path,"conf":conf,
                         "tp":tp,"tn":tn,"fp":fp,"fn":fn,"total":total,
                         "accuracy":round(acc,6),"precision":round(prec,6),
                         "recall":round(rec,6),"f1_score":round(f1,6)})
            print(f"  {model_name} conf={conf:.2f}: TP={tp} TN={tn} FP={fp} FN={fn} F1={f1:.4f}")

    csv_path = REPORT_DIR / "yolo_model_comparison.csv"
    fields = ["model","model_path","conf","tp","tn","fp","fn","total","accuracy","precision","recall","f1_score"]
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

    best = max(rows, key=lambda x: x["f1_score"])
    rec80 = [r for r in rows if r["recall"]>=0.8]
    best_rec80 = max(rec80, key=lambda x: x["precision"]) if rec80 else None

    print(f"\nF1最佳: {best['model']} conf={best['conf']:.2f} TP={best['tp']} TN={best['tn']} FP={best['fp']} FN={best['fn']} Prec={best['precision']*100:.1f}% Rec={best['recall']*100:.1f}% F1={best['f1_score']:.4f}")
    if best_rec80:
        print(f"Recall>=80%最佳: {best_rec80['model']} conf={best_rec80['conf']:.2f} Prec={best_rec80['precision']*100:.1f}% Rec={best_rec80['recall']*100:.1f}% F1={best_rec80['f1_score']:.4f}")
    else:
        print("Recall>=80%: 无方法达到")
    print(f"\n报表: {csv_path}")
    print("="*60)

if __name__=="__main__":
    main()
