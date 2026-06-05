"""YOLO Segmentation 训练脚本

使用 ultralytics YOLO segmentation 模型训练缺陷检测。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    print("=" * 60)
    print("YOLO Segmentation 训练")
    print("=" * 60)

    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"ultralytics 版本：{ultralytics.__version__}")
    except ImportError:
        print("\nultralytics 未安装。")
        print("请先安装依赖：")
        print("  pip install ultralytics")
        print("\n安装后运行：")
        print("  python scripts/train_yolo_seg.py")
        print("=" * 60)
        return

    data_yaml = Path("data/yolo_seg/data.yaml")
    if not data_yaml.exists():
        print(f"\n错误：{data_yaml} 不存在")
        print("请先运行：python scripts/prepare_yolo_seg_dataset.py")
        print("=" * 60)
        return

    print(f"\n数据配置：{data_yaml}")
    print("模型：yolov8n-seg.pt（YOLOv8 nano segmentation 预训练权重）")
    print("训练参数：epochs=50, imgsz=640, batch=8")
    print("\n开始训练...")

    model = YOLO("yolov8n-seg.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        name="yolov8n_seg_defect",
        project="outputs/yolo_seg/runs",
        exist_ok=True
    )

    print(f"\n训练完成。模型保存至：outputs/yolo_seg/runs/yolov8n_seg_defect/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
