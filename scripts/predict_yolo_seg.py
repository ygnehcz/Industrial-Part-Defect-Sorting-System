"""YOLO Segmentation 推理脚本"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    print("=" * 60)
    print("YOLO Segmentation 推理")
    print("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\nultralytics 未安装。")
        print("请先安装依赖：")
        print("  pip install ultralytics")
        print("=" * 60)
        return

    best_pt = Path("outputs/yolo_seg/runs/yolov8n_seg_defect/weights/best.pt")
    if not best_pt.exists():
        print(f"\n训练模型不存在：{best_pt}")
        print("请先运行训练：python scripts/train_yolo_seg.py")
        print("=" * 60)
        return

    val_dir = Path("data/yolo_seg/images/val")
    if not val_dir.exists():
        print(f"\nval 图片目录不存在：{val_dir}")
        print("=" * 60)
        return

    pred_dir = Path("outputs/yolo_seg/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n模型：{best_pt}")
    print(f"推理目录：{val_dir}")

    model = YOLO(str(best_pt))
    results = model.predict(
        source=str(val_dir),
        save=True,
        project=str(pred_dir.parent),
        name="predictions",
        exist_ok=True
    )

    total = len(results)
    print(f"\n推理完成：{total} 张图片")
    print(f"结果已保存至：{pred_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
