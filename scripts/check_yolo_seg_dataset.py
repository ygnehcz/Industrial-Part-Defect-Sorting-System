"""YOLO Segmentation 数据集检查脚本"""

import sys
import csv
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

YOLO_DIR = Path("data/yolo_seg")
REPORT_DIR = Path("outputs/yolo_seg/reports")


def check_label_format(label_path, img_w, img_h):
    """检查单个标签文件格式"""
    issues = []
    try:
        with open(label_path, "r") as f:
            content = f.read().strip()
    except Exception as e:
        return [f"读取失败: {e}"]

    if not content:
        return []  # Empty label is valid

    for line_no, line in enumerate(content.split("\n"), 1):
        parts = line.strip().split()
        if len(parts) < 7:
            issues.append(f"行{line_no}: 点数不足 (最少3对坐标)")
            continue
        if parts[0] != "0":
            issues.append(f"行{line_no}: 类别ID应为0，实际={parts[0]}")
        coords = parts[1:]
        if len(coords) % 2 != 0:
            issues.append(f"行{line_no}: 坐标数量不是偶数")
            continue
        for i in range(0, len(coords), 2):
            x, y = float(coords[i]), float(coords[i+1])
            if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
                issues.append(f"行{line_no}: 坐标越界 ({x}, {y})")

    return issues


def main():
    print("=" * 60)
    print("YOLO Segmentation 数据集检查")
    print("=" * 60)

    issues_found = []

    # Check directory structure
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        p = YOLO_DIR / sub
        if p.exists():
            count = len(list(p.iterdir()))
            print(f"  [OK] {sub} 存在 ({count} 文件)")
        else:
            print(f"  [FAIL] {sub} 不存在")
            issues_found.append(f"missing: {sub}")

    # Check data.yaml
    yaml_path = YOLO_DIR / "data.yaml"
    if yaml_path.exists():
        print(f"  [OK] data.yaml 存在")
    else:
        print(f"  [FAIL] data.yaml 不存在")
        issues_found.append("missing: data.yaml")

    # Check image-label pairing
    stats_rows = []
    for split_name in ["train", "val"]:
        img_dir = YOLO_DIR / "images" / split_name
        lbl_dir = YOLO_DIR / "labels" / split_name
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        img_files = sorted(img_dir.glob("*.png"))
        lbl_files = {f.stem: f for f in lbl_dir.glob("*.txt")}

        total = len(img_files)
        paired = 0
        unpaired_imgs = 0
        unpaired_lbls = len(lbl_files) - total
        empty_lbls = 0
        nonempty_lbls = 0
        label_issues = 0

        for img_path in img_files:
            stem = img_path.stem
            if stem in lbl_files:
                paired += 1
                lbl_path = lbl_files[stem]
                content = lbl_path.read_text().strip()
                if not content:
                    empty_lbls += 1
                else:
                    nonempty_lbls += 1
            else:
                unpaired_imgs += 1

        # Random sample check
        sample_imgs = random.sample(img_files, min(5, len(img_files)))
        for img_path in sample_imgs:
            stem = img_path.stem
            if stem in lbl_files:
                import cv2
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    lbl_iss = check_label_format(lbl_files[stem], w, h)
                    if lbl_iss:
                        label_issues += len(lbl_iss)

        print(f"\n  {split_name}: images={total}, paired={paired}, "
              f"empty={empty_lbls}, nonempty={nonempty_lbls}, label_issues={label_issues}")

        stats_rows.append({
            "split": split_name, "total_images": total, "paired_labels": paired,
            "empty_labels": empty_lbls, "nonempty_labels": nonempty_lbls,
            "label_format_issues": label_issues
        })

    # Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "yolo_seg_dataset_check.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split","total_images","paired_labels","empty_labels","nonempty_labels","label_format_issues"])
        w.writeheader()
        w.writerows(stats_rows)

    print(f"\n检查报表：{report_path}")

    if issues_found:
        print(f"\n发现 {len(issues_found)} 个问题：")
        for iss in issues_found:
            print(f"  - {iss}")
    else:
        print("\n数据集检查通过。")

    print("=" * 60)


if __name__ == "__main__":
    main()
