import cv2
import numpy as np


def apply_open_operation(binary_image, kernel_size):
    """开运算：先腐蚀后膨胀，用于去除小噪声点"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    return opened


def apply_close_operation(binary_image, kernel_size):
    """闭运算：先膨胀后腐蚀，用于填补缺陷区域断裂"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return closed


def apply_morphology(binary_image, kernel_size, use_open=True, use_close=True):
    """组合形态学处理

    先开运算去噪，再闭运算填补断裂。
    可通过 use_open / use_close 控制是否启用对应操作。
    """
    result = binary_image
    if use_open:
        result = apply_open_operation(result, kernel_size)
    if use_close:
        result = apply_close_operation(result, kernel_size)
    return result
