from pathlib import Path

import cv2
import numpy as np


def main():
    output_dir = Path("data/samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((300, 300, 3), dtype=np.uint8)

    cv2.rectangle(image, (80, 80), (220, 220), (80, 80, 80), -1)

    cv2.circle(image, (150, 150), 25, (255, 255, 255), -1)

    output_path = output_dir / "sample_part_01.jpg"
    cv2.imwrite(str(output_path), image)

    print(f"测试图片已生成：{output_path}")


if __name__ == "__main__":
    main()