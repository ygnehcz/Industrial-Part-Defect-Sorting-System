"""传统视觉增强方法模块

提供 CLAHE、Top-hat、Black-hat、Sobel、Canny 等增强方法，
以及连通域特征过滤。
"""

import cv2
import numpy as np


def apply_clahe(gray_image, clip_limit=2.0, tile_grid_size=8):
    """CLAHE（对比度受限自适应直方图均衡化）

    增强局部对比度，对光照不均匀的场景有帮助。
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(gray_image)


def apply_tophat(gray_image, kernel_size=15):
    """Top-hat（顶帽变换）

    原图 - 开运算结果，提取比周围亮的细小区域。
    适合检测亮缺陷。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)


def apply_blackhat(gray_image, kernel_size=15):
    """Black-hat（黑帽变换）

    闭运算结果 - 原图，提取比周围暗的细小区域。
    适合检测暗缺陷。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)


def apply_sobel(gray_image):
    """Sobel 边缘检测（梯度幅值）

    突出边缘和纹理变化区域，缺陷通常表现为异常边缘。
    """
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    return magnitude


def apply_canny(gray_image, low_threshold=50, high_threshold=150):
    """Canny 边缘检测

    返回二值边缘图。
    """
    return cv2.Canny(gray_image, low_threshold, high_threshold)


def filter_contours_by_features(contours, min_area=50, max_area=None,
                                 min_aspect_ratio=None, max_aspect_ratio=None):
    """按几何特征过滤轮廓

    参数：
    - min_area: 最小面积
    - max_area: 最大面积
    - min_aspect_ratio: 最小长宽比 (w/h)
    - max_aspect_ratio: 最大长宽比
    """
    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        if min_aspect_ratio is not None or max_aspect_ratio is not None:
            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / h if h > 0 else 0
            if min_aspect_ratio is not None and ar < min_aspect_ratio:
                continue
            if max_aspect_ratio is not None and ar > max_aspect_ratio:
                continue
        filtered.append(cnt)
    return filtered
