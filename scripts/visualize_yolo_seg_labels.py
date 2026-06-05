"""YOLO Segmentation 标注可视化脚本"""

import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

YOLO_DIR = Path("data/yolo_seg")
OUTPUT_DIR = Path("docs/images/yolo_seg_samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_yolo_seg(image, label_path):
    """将 YOLO segmentation label 画到图像上"""
    h, w = image.shape[:2]
    overlay = image.copy()

    if not label_path.exists():
        return image

    content = label_path.read_text().strip()
    if not content:
        return image

    for line in content.split("\n"):
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = [float(x) for x in parts[1:]]
        points = []
        for i in range(0, len(coords), 2):
            px = int(coords[i] * w)
            py = int(coords[i+1] * h)
            points.append([px, py])
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        cv2.polylines(image, [pts], True, (0, 0, 255), 2)

    cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)
    return image


def main():
    print("=" * 60)
    print("YOLO Segmentation 标注可视化")
    print("=" * 60)

    for split_name in ["train", "val"]:
        img_dir = YOLO_DIR / "images" / split_name
        lbl_dir = YOLO_DIR / "labels" / split_name
        if not img_dir.exists():
            continue

        # Find NG samples (non-empty labels)
        ng_samples = []
        ok_samples = []
        for img_path in sorted(img_dir.glob("*.png")):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists() and lbl_path.read_text().strip():
                ng_samples.append((img_path, lbl_path))
            else:
                ok_samples.append((img_path, lbl_path))

        # Save NG samples
        selected_ng = random.sample(ng_samples, min(3, len(ng_samples)))
        for i, (img_path, lbl_path) in enumerate(selected_ng):
            img = cv2.imread(str(img_path))
            if img is None: continue
            drawn = draw_yolo_seg(img, lbl_path)
            out = OUTPUT_DIR / f"{split_name}_ng_sample_{i+1:02d}.png"
            cv2.imwrite(str(out), drawn)
            print(f"  [NG] {out.name}")

        # Save 1 OK sample
        if ok_samples:
            img_path, lbl_path = random.choice(ok_samples)
            img = cv2.imread(str(img_path))
            if img is not None:
                drawn = draw_yolo_seg(img, lbl_path)
                out = OUTPUT_DIR / f"{split_name}_ok_sample_01.png"
                cv2.imwrite(str(out), drawn)
                print(f"  [OK] {out.name} (no defect)")

    files = sorted(OUTPUT_DIR.glob("*.png"))
    print(f"\n共生成 {len(files)} 张可视化图")
    print(f"保存目录：{OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
