"""YOLO 10-epoch 模型评价脚本"""

import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2, numpy as np

BEST_PT = Path("outputs/yolo_seg/runs/yolov8n_seg_10epochs/weights/best.pt")
VAL_IMG_DIR = Path("data/yolo_seg/images/val")
GT_MASK_DIR = Path(r"D:\Industrial-Surface-Defect-Inspection\data\raw\test")
REPORT_DIR = Path("outputs/yolo_seg/reports")

def get_true_status(mask_path):
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt is None: return "UNKNOWN", 0
    area = int(np.count_nonzero(gt))
    return ("NG" if area>0 else "OK"), area

def main():
    print("="*60)
    print("YOLO 10-epoch 模型评价")
    print("="*60)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(str(BEST_PT))
    img_files = sorted(VAL_IMG_DIR.glob("*.png"))
    results = []
    tp=tn=fp=fn=0

    for img_path in img_files:
        stem = img_path.stem
        gt_path = GT_MASK_DIR / f"{stem}_GT.png"
        true_status, gt_area = get_true_status(gt_path)

        pred = model(str(img_path), conf=0.05, verbose=False)
        has_mask = pred and pred[0].masks is not None
        pred_status = "NG" if has_mask else "OK"

        pred_area = 0; iou = dice = 0.0
        if has_mask:
            masks = pred[0].masks.data.cpu().numpy()
            h,w = pred[0].orig_shape
            pm = np.zeros((h,w), dtype=np.uint8)
            for m in masks:
                pm = np.maximum(pm, cv2.resize(m.astype(np.uint8),(w,h)))
            pred_area = int(np.count_nonzero(pm))
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                gt = cv2.resize(gt, (w,h))
                gb = (gt>0).astype(np.uint8); pb = (pm>0).astype(np.uint8)
                inter = np.sum(gb & pb); union = np.sum(gb | pb)
                iou = inter/union if union>0 else 0
                dice = 2*inter/(np.sum(gb)+np.sum(pb)) if (np.sum(gb)+np.sum(pb))>0 else 0

        rt = "UNKNOWN"
        if true_status=="NG" and pred_status=="NG": rt="TP"; tp+=1
        elif true_status=="OK" and pred_status=="OK": rt="TN"; tn+=1
        elif true_status=="OK" and pred_status=="NG": rt="FP"; fp+=1
        elif true_status=="NG" and pred_status=="OK": rt="FN"; fn+=1

        results.append({"image_name":img_path.name,"true_status":true_status,"pred_status":pred_status,
                        "result_type":rt,"gt_area":gt_area,"pred_area":pred_area,"iou":round(iou,6),"dice":round(dice,6)})

    total=len(results)
    acc=(tp+tn)/total; prec=tp/(tp+fp) if(tp+fp)>0 else 0; rec=tp/(tp+fn) if(tp+fn)>0 else 0
    f1=2*prec*rec/(prec+rec) if(prec+rec)>0 else 0

    tp_r=[r for r in results if r["result_type"]=="TP"]
    ng_r=[r for r in results if r["true_status"]=="NG"]
    miou_tp=sum(r["iou"] for r in tp_r)/len(tp_r) if tp_r else 0
    mdice_tp=sum(r["dice"] for r in tp_r)/len(tp_r) if tp_r else 0
    miou_ng=sum(r["iou"] for r in ng_r)/len(ng_r) if ng_r else 0
    mdice_ng=sum(r["dice"] for r in ng_r)/len(ng_r) if ng_r else 0

    print(f"\nTP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Acc={acc*100:.1f}%, Prec={prec*100:.1f}%, Rec={rec*100:.1f}%, F1={f1:.4f}")
    print(f"mean IoU(TP)={miou_tp:.4f}, mean Dice(TP)={mdice_tp:.4f}")
    print(f"mean IoU(NG)={miou_ng:.4f}, mean Dice(NG)={mdice_ng:.4f}")

    detail_path = REPORT_DIR / "yolo_seg_10epochs_eval_report.csv"
    with open(detail_path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["image_name","true_status","pred_status","result_type","gt_area","pred_area","iou","dice"])
        w.writeheader(); w.writerows(results)

    summary_path = REPORT_DIR / "yolo_seg_10epochs_summary.csv"
    with open(summary_path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["tp","tn","fp","fn","total","accuracy","precision","recall","f1","mean_iou_tp","mean_dice_tp","mean_iou_ng","mean_dice_ng"])
        w.writeheader(); w.writerow({"tp":tp,"tn":tn,"fp":fp,"fn":fn,"total":total,"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"mean_iou_tp":miou_tp,"mean_dice_tp":mdice_tp,"mean_iou_ng":miou_ng,"mean_dice_ng":mdice_ng})

    print(f"\n报表: {detail_path}")
    print(f"     : {summary_path}")
    print("="*60)

if __name__=="__main__":
    main()
