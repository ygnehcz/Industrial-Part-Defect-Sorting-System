import cv2, csv, numpy as np
from pathlib import Path
from ultralytics import YOLO

VAL_IMG = Path("data/yolo_seg/images/val")
VAL_LBL = Path("data/yolo_seg/labels/val")
GT_DIR = Path(r"D:\Industrial-Surface-Defect-Inspection\data\raw\test")
OUT = Path("docs/images/yolo_model_comparison_cases")
OUT.mkdir(parents=True, exist_ok=True)

model22 = YOLO("outputs/yolo_seg/runs/yolov8n_seg_smoke/weights/best.pt")
model23 = YOLO("outputs/yolo_seg/runs/yolov8n_seg_10epochs/weights/best.pt")
conf = 0.05

def get_true(lp):
    return "NG" if (lp.exists() and lp.read_text().strip()) else "OK"

def pred_status(model, img_path, conf):
    r = model(str(img_path), conf=conf, verbose=False)
    return 0 if (r and r[0].masks is not None) else 0  # Always 0 for simplicity - just checking if masks exist

# Actually let me do it properly
def get_pred(model, img_path, conf):
    r = model(str(img_path), conf=conf, verbose=False)
    return "NG" if (r and r[0].masks is not None) else "OK"

def make_compare(img_name, out_name):
    img = cv2.imread(str(VAL_IMG / img_name))
    if img is None: return False
    stem = Path(img_name).stem
    gt = cv2.imread(str(GT_DIR / f"{stem}_GT.png"))
    if gt is None: return False
    if len(gt.shape)==3: gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    gt_c = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
    
    h,w = img.shape[:2]
    def make_mask(model):
        pred = model(str(VAL_IMG/img_name), conf=conf, verbose=False)
        pm = np.zeros((h,w,3), dtype=np.uint8)
        if pred and pred[0].masks is not None:
            for m in pred[0].masks.data.cpu().numpy():
                r = cv2.resize(m.astype(np.uint8),(w,h))
                pm[:,:,2] = np.maximum(pm[:,:,2], r*255)
        return cv2.addWeighted(img,0.5,pm,0.5,0)
    
    m22 = make_mask(model22); m23 = make_mask(model23)
    hmax = max(img.shape[0], gt.shape[0])
    panels = []
    for label, panel in [("Original",img),("GT Mask",gt_c),("V2.2 3ep",m22),("V2.3 10ep",m23)]:
        if panel.shape[0]<hmax:
            panel = cv2.copyMakeBorder(panel,0,hmax-panel.shape[0],0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
        cv2.putText(panel,label,(5,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
        panels.append(panel)
    cv2.imwrite(str(OUT/out_name), np.hstack(panels))
    return True

img_files = sorted(VAL_IMG.glob("*.png"))
cases = {"both_TP":[], "both_FN":[], "v22_better":[], "v23_better":[]}

for img_path in img_files:
    true = get_true(VAL_LBL / f"{img_path.stem}.txt")
    p22 = get_pred(model22, img_path, conf)
    p23 = get_pred(model23, img_path, conf)
    if true=="NG" and p22=="NG" and p23=="NG": cases["both_TP"].append(img_path.name)
    elif true=="NG" and p22=="OK" and p23=="OK": cases["both_FN"].append(img_path.name)
    elif true=="NG" and p22=="NG" and p23=="OK": cases["v22_better"].append(img_path.name)
    elif true=="NG" and p22=="OK" and p23=="NG": cases["v23_better"].append(img_path.name)

for cat, files in cases.items():
    for i, fn in enumerate(files[:2]):
        ok = make_compare(fn, f"{cat}_{i+1:02d}.png")
        print(f"{cat}: {fn} - {'OK' if ok else 'SKIP'}")

print(f"\nTotal: {len(list(OUT.glob('*.png')))} files")
