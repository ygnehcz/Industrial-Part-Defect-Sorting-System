"""YOLO Segmentation Demo Entry Point"""

import sys
import cv2
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

BEST_PT = Path("outputs/yolo_seg/runs/yolov8n_seg_smoke/weights/best.pt")
VAL_IMG = Path("data/yolo_seg/images/val")
DEMO_DIR = Path("outputs/demo")
CONF = 0.05
DEMO_COUNT = 8


def main():
    print("=" * 50)
    print("Industrial Surface Defect Detection - YOLO Demo")
    print("=" * 50)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\nultralytics not installed. Run: pip install ultralytics")
        return

    if not BEST_PT.exists():
        print(f"\nModel not found: {BEST_PT}")
        print("Run YOLO training first or download best.pt from releases.")
        return

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(BEST_PT))
    print(f"\nModel: {BEST_PT}")
    print(f"Confidence: {CONF}")

    img_files = sorted(VAL_IMG.glob("*.png"))
    from src.gt_mask_utils import find_image_mask_pairs, get_true_status_from_mask
    gt_dir = Path(r"D:\Industrial-Surface-Defect-Inspection\data\raw\test")
    ng_imgs, ok_imgs = [], []
    for f in img_files[:300]:
        gt = gt_dir / f"{f.stem}_GT.png"
        if gt.exists():
            status = get_true_status_from_mask(gt)
            if status == "NG": ng_imgs.append(f)
            else: ok_imgs.append(f)
    
    demo_imgs = (ng_imgs[:DEMO_COUNT//2] + ok_imgs[:DEMO_COUNT//2])[:DEMO_COUNT]
    print(f"\nProcessing {len(demo_imgs)} demo images...\n")

    ng_pred = ok_pred = 0
    for img_path in demo_imgs:
        result = model(str(img_path), conf=CONF, verbose=False)
        has_defect = result and result[0].masks is not None
        status = "NG" if has_defect else "OK"
        if has_defect: ng_pred += 1
        else: ok_pred += 1
        
        out_img = result[0].plot() if has_defect else cv2.imread(str(img_path))
        out_path = DEMO_DIR / f"{img_path.stem}_demo_{status}.jpg"
        cv2.imwrite(str(out_path), out_img)
        print(f"  {img_path.name} -> {status}")

    print(f"\nResults: NG={ng_pred}, OK={ok_pred}")
    print(f"Output: {DEMO_DIR.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
