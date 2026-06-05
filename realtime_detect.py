import cv2
import time
from pathlib import Path
from datetime import datetime

from config import (
    CAMERA_INDEX,
    GRAY_THRESHOLD,
    THRESHOLD,
    MIN_CONTOUR_AREA,
    MORPH_KERNEL_SIZE,
    USE_OPEN_OPERATION,
    USE_CLOSE_OPERATION,
    REALTIME_SAVE_DIR,
    SAVE_NG_FRAME
)
from src.image_processor import (
    convert_to_gray,
    threshold_image,
    calculate_defect_area,
    find_defect_contours,
    draw_defect_boxes
)
from src.morphology import apply_morphology
from src.judge import judge_status


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"错误：无法打开摄像头 (index={CAMERA_INDEX})")
        print("请确认：")
        print("  1. 摄像头已连接")
        print("  2. 没有被其他程序占用")
        print("  3. 尝试修改 config.py 中的 CAMERA_INDEX")
        return

    print("摄像头已打开，按 q 退出")

    save_dir = Path(REALTIME_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    prev_status = None
    frame_count = 0
    ng_save_interval = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        frame_count += 1

        gray = convert_to_gray(frame)

        binary = threshold_image(gray, GRAY_THRESHOLD)

        morph = apply_morphology(
            binary,
            MORPH_KERNEL_SIZE,
            use_open=USE_OPEN_OPERATION,
            use_close=USE_CLOSE_OPERATION
        )

        defect_area = calculate_defect_area(morph)

        status = judge_status(defect_area, THRESHOLD)

        contours = find_defect_contours(morph)

        display_frame, defect_count = draw_defect_boxes(
            frame, contours, MIN_CONTOUR_AREA
        )

        color = (0, 255, 0) if status == "OK" else (0, 0, 255)

        cv2.putText(
            display_frame,
            f"Status: {status}  Area: {defect_area}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        if SAVE_NG_FRAME and status == "NG":
            if prev_status != "NG" or frame_count % ng_save_interval == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = save_dir / f"ng_frame_{timestamp}.jpg"
                cv2.imwrite(str(save_path), display_frame)

        prev_status = status

        cv2.imshow("Industrial Part Defect Detection (q to quit)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("摄像头已关闭")


if __name__ == "__main__":
    main()
