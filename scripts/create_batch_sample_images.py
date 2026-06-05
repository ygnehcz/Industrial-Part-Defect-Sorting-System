from pathlib import Path

import cv2
import numpy as np


def create_base_image():
    image = np.zeros((300, 300, 3), dtype=np.uint8)

    # 画一个灰色零件区域
    cv2.rectangle(image, (80, 80), (220, 220), (80, 80, 80), -1)

    return image


def main():
    output_dir = Path("data/samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # OK 图片：没有明显白色缺陷
    ok_image_01 = create_base_image()
    cv2.imwrite(str(output_dir / "part_001_OK.jpg"), ok_image_01)

    ok_image_02 = create_base_image()
    cv2.circle(ok_image_02, (150, 150), 3, (255, 255, 255), -1)
    cv2.imwrite(str(output_dir / "part_002_OK.jpg"), ok_image_02)

    # NG 图片：有明显白色缺陷
    ng_image_01 = create_base_image()
    cv2.circle(ng_image_01, (150, 150), 20, (255, 255, 255), -1)
    cv2.imwrite(str(output_dir / "part_003_NG.jpg"), ng_image_01)

    ng_image_02 = create_base_image()
    cv2.rectangle(ng_image_02, (130, 130), (190, 170), (255, 255, 255), -1)
    cv2.imwrite(str(output_dir / "part_004_NG.jpg"), ng_image_02)

    ng_image_03 = create_base_image()
    cv2.circle(ng_image_03, (120, 120), 15, (255, 255, 255), -1)
    cv2.circle(ng_image_03, (180, 180), 15, (255, 255, 255), -1)
    cv2.imwrite(str(output_dir / "part_005_NG.jpg"), ng_image_03)

    print(f"批量测试图片已生成：{output_dir}")


if __name__ == "__main__":
    main()