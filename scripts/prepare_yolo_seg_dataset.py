"""YOLO Segmentation 数据集转换脚本

将 KolektorSDD2 风格 *_GT.png mask 转换为 YOLO segmentation polygon 格式。
"""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

RAW_DIR = Path(r"D:\Industrial-Surface-Defect-Inspection\data\raw")
OUTPUT_DIR = Path("data/yolo_seg")
MIN_CONTOUR_AREA = 5
APPROX_EPSILON = 0.001


def mask_to_yolo_seg(mask_path, img_w, img_h):
    """将 GT mask 转换为 YOLO segmentation label 行列表"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        # Simplify polygon
        epsilon = APPROX_EPSILON * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Normalize to 0-1
        points = []
        for pt in approx:
            x = pt[0][0] / img_w
            y = pt[0][1] / img_h
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            points.append(f"{x:.6f} {y:.6f}")

        if len(points) >= 3:
            lines.append(f"0 " + " ".join(points))

    return lines


def convert_split(split_name):
    """转换一个数据划分（train 或 val，val 使用原始 test 集）"""
    src_name = "test" if split_name == "val" else split_name
    src_dir = RAW_DIR / src_name

    img_out = OUTPUT_DIR / "images" / split_name
    lbl_out = OUTPUT_DIR / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    png_files = sorted(src_dir.glob("*.png"))
    images = [f for f in png_files if "_GT" not in f.stem]

    total = 0
    ng_count = 0
    empty_labels = 0
    nonempty_labels = 0
    failed = 0

    for img_path in images:
        total += 1
        mask_path = src_dir / f"{img_path.stem}_GT.png"
        if not mask_path.exists():
            failed += 1
            continue

        # Copy image
        shutil.copy2(str(img_path), str(img_out / img_path.name))

        img = cv2.imread(str(img_path))
        if img is None:
            failed += 1
            continue
        h, w = img.shape[:2]

        lines = mask_to_yolo_seg(mask_path, w, h)

        label_path = lbl_out / f"{img_path.stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")
                nonempty_labels += 1
                ng_count += 1
            else:
                f.write("")
                empty_labels += 1

        if total % 500 == 0:
            print(f"  {split_name}: {total}/{len(images)}")

    return {
        "split": split_name,
        "total_images": total,
        "ng_count": ng_count,
        "empty_labels": empty_labels,
        "nonempty_labels": nonempty_labels,
        "failed": failed
    }


def main():
    print("=" * 60)
    print("YOLO Segmentation 数据集转换")
    print("=" * 60)

    # Clean output dir
    if OUTPUT_DIR.exists():
        shutil.rmtree(str(OUTPUT_DIR))
        print("已清理旧数据集")

    stats = []
    for split_name in ["train", "val"]:
        print(f"\n转换 {split_name}...")
        s = convert_split(split_name)
        stats.append(s)

    # Write data.yaml
    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_content = f"""path: data/yolo_seg
train: images/train
val: images/val
names:
  0: defect
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\ndata.yaml 已生成：{yaml_path}")

    # Summary
    print("\n" + "=" * 60)
    print("转换统计")
    print("=" * 60)
    for s in stats:
        print(f"{s['split']}: images={s['total_images']}, NG={s['ng_count']}, "
              f"nonempty_label={s['nonempty_labels']}, empty_label={s['empty_labels']}, failed={s['failed']}")

    total_img = sum(s["total_images"] for s in stats)
    total_ng = sum(s["ng_count"] for s in stats)
    print(f"\n总计：{total_img} 对图像（NG={total_ng}）")
    print(f"输出目录：{OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
