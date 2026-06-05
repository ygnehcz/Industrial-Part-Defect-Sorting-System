"""GT Mask 数据读取与统计工具

用于读取 KolektorSDD2 风格数据集：
- 原图为 *.png（非 *_GT.png）
- GT mask 为 *_GT.png
- mask 全黑 = OK，含白像素 = NG
"""

import cv2
import numpy as np
from pathlib import Path


def find_image_mask_pairs(dataset_dir):
    """从数据目录中找到原图与对应 *_GT.png mask 的配对

    返回列表，每项为 (image_path, mask_path)
    """
    dataset_dir = Path(dataset_dir)
    pairs = []

    all_files = list(dataset_dir.glob("*.png"))
    mask_files = {f.stem.replace("_GT", ""): f for f in all_files if f.stem.endswith("_GT")}

    for f in all_files:
        if f.stem.endswith("_GT"):
            continue
        stem = f.stem
        if stem in mask_files:
            pairs.append((f, mask_files[stem]))

    return pairs


def get_true_status_from_mask(mask_path):
    """根据 GT mask 判断真实标签

    mask 含白色像素 → NG，全黑 → OK
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return "UNKNOWN"
    if np.count_nonzero(mask) > 0:
        return "NG"
    return "OK"


def calculate_gt_defect_area(mask_path):
    """返回 mask 中白色像素数量（GT 缺陷面积）"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0
    return int(np.count_nonzero(mask))


def calculate_gt_defect_ratio(mask_path):
    """返回白色像素占比 = 缺陷面积 / 总像素"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0.0
    total_pixels = mask.shape[0] * mask.shape[1]
    if total_pixels == 0:
        return 0.0
    return np.count_nonzero(mask) / total_pixels


def summarize_dataset_by_mask(dataset_dir):
    """统计数据集的 OK/NG 数量及缺陷面积分布"""
    pairs = find_image_mask_pairs(dataset_dir)

    total = len(pairs)
    ok_count = 0
    ng_count = 0
    defect_areas = []

    for _, mask_path in pairs:
        status = get_true_status_from_mask(mask_path)
        area = calculate_gt_defect_area(mask_path)
        if status == "OK":
            ok_count += 1
        else:
            ng_count += 1
            if area > 0:
                defect_areas.append(area)

    summary = {
        "total_count": total,
        "ok_count": ok_count,
        "ng_count": ng_count,
        "defect_area_min": min(defect_areas) if defect_areas else 0,
        "defect_area_max": max(defect_areas) if defect_areas else 0,
        "defect_area_mean": sum(defect_areas) / len(defect_areas) if defect_areas else 0,
        "defect_ratio_mean": 0.0
    }

    if defect_areas:
        ratios = []
        for _, mask_path in pairs:
            status = get_true_status_from_mask(mask_path)
            if status == "NG":
                ratios.append(calculate_gt_defect_ratio(mask_path))
        summary["defect_ratio_mean"] = sum(ratios) / len(ratios) if ratios else 0.0

    return summary
