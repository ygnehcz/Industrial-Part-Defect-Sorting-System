"""
YOLO 对比框架脚本

说明：
- 当前 OpenCV 规则方法是项目的基线检测方案
- YOLO 是深度学习扩展方向，本脚本提供对比框架
- 真正可用的缺陷检测需要：
  1. 真实工业缺陷数据集
  2. 数据标注（缺陷类别 + 边界框）
  3. 模型训练（如 yolov8n.pt 微调）
  4. 验证和部署
- 当前脚本如 ultralytics 已安装，可用 yolov8n.pt 做通用目标检测对比
- 如 ultralytics 未安装，给出安装提示
"""
import csv
from pathlib import Path
from datetime import datetime
from config import SAMPLE_IMAGE_DIR, REPORT_DIR, IMAGE_OUTPUT_DIR

YOLO_OUTPUT_DIR = "outputs/yolo"
YOLO_COMPARE_CSV = "outputs/reports/yolo_compare_report.csv"


def check_ultralytics():
    try:
        import ultralytics
        return True, ultralytics.__version__
    except ImportError:
        return False, None


def load_opencv_results(report_path):
    """从已有的 OpenCV 检测报表中读取结果"""
    opencv_results = {}
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                opencv_results[row["image_name"]] = row
    except FileNotFoundError:
        pass
    return opencv_results


def run_yolo_inference():
    from ultralytics import YOLO

    yolo_output_dir = Path(YOLO_OUTPUT_DIR)
    yolo_output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)

    sample_dir = Path(SAMPLE_IMAGE_DIR)
    image_paths = sorted(sample_dir.glob("*.jpg"))

    if not image_paths:
        print("未找到测试图片")
        return

    print(f"加载 YOLOv8n 预训练模型...")
    model = YOLO("yolov8n.pt")

    opencv_report_path = report_dir / "batch_detection_report.csv"
    opencv_results = load_opencv_results(opencv_report_path)

    compare_rows = []

    for image_path in image_paths:
        image_name = image_path.name
        print(f"YOLO 推理：{image_name}")

        results = model(str(image_path))

        yolo_detected = 0
        for result in results:
            yolo_detected += len(result.boxes) if result.boxes is not None else 0

        save_path = yolo_output_dir / f"{image_path.stem}_yolo.jpg"
        if results:
            results[0].save(filename=str(save_path))

        opencv_info = opencv_results.get(image_name, {})
        opencv_pred = opencv_info.get("pred_status", "N/A")

        compare_rows.append({
            "image_name": image_name,
            "opencv_pred_status": opencv_pred,
            "yolo_detected": yolo_detected,
            "yolo_result_path": str(save_path),
            "note": "YOLO 通用目标检测结果，非工业缺陷专用模型"
        })

    compare_path = Path(YOLO_COMPARE_CSV)
    fieldnames = [
        "image_name",
        "opencv_pred_status",
        "yolo_detected",
        "yolo_result_path",
        "note"
    ]
    with open(compare_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(compare_rows)

    print(f"\nYOLO 对比报表已保存：{compare_path}")
    print(f"YOLO 结果图保存目录：{yolo_output_dir}")

    print("\n====== 对比摘要 ======")
    for row in compare_rows:
        print(
            f"{row['image_name']}: "
            f"OpenCV={row['opencv_pred_status']}, "
            f"YOLO检测对象数={row['yolo_detected']}"
        )


def main():
    print("=" * 50)
    print("OpenCV vs YOLO 缺陷检测对比框架")
    print("=" * 50)

    has_ultralytics, version = check_ultralytics()

    if has_ultralytics:
        print(f"\nultralytics 已安装 (版本: {version})")
        print("注意：当前使用 yolov8n.pt 通用预训练模型")
        print("该模型用于 COCO 类别目标检测，非工业缺陷专用模型")
        print("真正的工业缺陷检测需要：")
        print("  1. 收集并标注真实缺陷图片数据集")
        print("  2. 使用 YOLO 进行迁移学习/微调")
        print("  3. 在验证集上评估模型性能")
        print()
        run_yolo_inference()
    else:
        print("\nultralytics 未安装")
        print("YOLO 为可选扩展依赖，如需使用请运行：")
        print("  pip install ultralytics")
        print()
        print("安装后可运行本脚本进行 OpenCV 与 YOLO 的对比：")
        print("  python yolo_compare.py")
        print()
        print("当前项目以 OpenCV 传统规则方法为基线，")
        print("YOLO 是后续深度学习扩展方向。")


if __name__ == "__main__":
    main()
