"""Auto-generate project demo video using cv2.VideoWriter"""

import cv2
import numpy as np
from pathlib import Path

W, H = 1280, 720
FPS = 24
SEC_PER_PAGE = 5
OUT_PATH = Path("outputs/demo/project_demo_v3.mp4")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

IMG_BASE = Path("docs/images")

def text_page(lines, bg_color=(30,30,40), text_color=(255,255,255)):
    """Create a text-only page frame"""
    frame = np.full((H,W,3), bg_color, dtype=np.uint8)
    y_start = H // 2 - len(lines) * 25
    for i, line in enumerate(lines):
        font_scale = 1.2 if i == 0 else 0.7
        thickness = 2 if i == 0 else 1
        size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        x = (W - size[0]) // 2
        y = y_start + i * 45
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    return frame

def image_page(img_path, title="", subtitle=""):
    """Create a page with an image centered"""
    frame = np.full((H,W,3), (30,30,40), dtype=np.uint8)
    if not img_path.exists():
        cv2.putText(frame, f"Image not found: {img_path.name}", (100,H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 1)
        return frame

    img = cv2.imread(str(img_path))
    if img is None:
        cv2.putText(frame, "Failed to load image", (100,H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 1)
        return frame

    # Scale to fit
    ih, iw = img.shape[:2]
    scale = min((W-80)/iw, (H-120)/ih)
    nw, nh = int(iw*scale), int(ih*scale)
    img = cv2.resize(img, (nw, nh))
    x, y = (W-nw)//2, 60
    frame[y:y+nh, x:x+nw] = img

    if title:
        cv2.putText(frame, title, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    if subtitle:
        cv2.putText(frame, subtitle, (50, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    return frame

def metrics_page(title, metrics):
    """Text page with metrics table"""
    lines = [title, ""] + metrics
    return text_page(lines)

def generate():
    print("Generating demo video...")
    
    # Try different codecs
    codecs = ["mp4v", "avc1", "XVID"]
    out = None
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, (W, H))
        if writer.isOpened():
            out = writer
            print(f"  Using codec: {codec}")
            break
        writer.release()

    if out is None:
        print("ERROR: Could not open VideoWriter with any codec")
        return

    frames_per_page = SEC_PER_PAGE * FPS

    # --- Page 1: Title ---
    frame = text_page([
        "Industrial Surface Defect Detection System",
        "OpenCV Baseline + YOLOv8 Segmentation",
        "",
        "Real Industrial Dataset  |  3335 Image-Mask Pairs"
    ])
    for _ in range(frames_per_page):
        out.write(frame)

    # --- Page 2: Project Roadmap ---
    frame = text_page([
        "Project Roadmap",
        "",
        "V1.x - Real Dataset + OpenCV CLAHE Enhancement",
        "V2.x - YOLOv8-seg Training & Model Selection",
        "V3.x - Demo & Deployment"
    ])
    for _ in range(frames_per_page):
        out.write(frame)

    # --- Page 3: Dataset ---
    frame = text_page([
        "Dataset Overview",
        "",
        "Train: 2331 pairs  |  Val: 1004 pairs",
        "OK: 2979  |  NG: 356",
        "Pixel-level GT mask available (*_GT.png)"
    ])
    for _ in range(frames_per_page):
        out.write(frame)

    # --- Page 4: OpenCV CLAHE Baseline ---
    frame = metrics_page("V1.2 - OpenCV CLAHE Baseline", [
        "Method: CLAHE + Fixed Threshold + Morphology",
        "",
        "Precision = 82.3%",
        "Recall    = 59.1%",
        "F1 Score  = 0.688",
        "",
        "TP=65  TN=880  FP=14  FN=45"
    ])
    for _ in range(frames_per_page):
        out.write(frame)

    # --- Page 5: YOLO Best ---
    frame = metrics_page("V2.2 - YOLOv8n-seg (3 epochs, conf=0.05)", [
        "Best Model - Highest F1",
        "",
        "Precision = 90.4%",
        "Recall    = 77.3%",
        "F1 Score  = 0.833",
        "",
        "TP=85  TN=885  FP=9  FN=25"
    ])
    for _ in range(frames_per_page):
        out.write(frame)

    # --- Page 6: Recall Priority ---
    frame = metrics_page("Recall-Priority Option (conf=0.03)", [
        "When minimizing FN is critical:",
        "",
        "Precision = 85.6%",
        "Recall    = 80.9%",
        "F1 Score  = 0.832",
        "",
        "TP=89  TN=879  FP=15  FN=21"
    ])
    for _ in range(frames_per_page):
        out.write(frame)

    # --- Chart pages ---
    charts = [
        (IMG_BASE / "experiment_charts/v1_1_vs_v1_2_metrics.png", "OpenCV CLAHE vs Baseline Comparison"),
        (IMG_BASE / "yolo_training_charts/yolo_f1_comparison.png", "F1 Score Across Versions"),
        (IMG_BASE / "yolo_model_selection_charts/model_conf_precision_recall_curve.png", "Precision-Recall Curve"),
        (IMG_BASE / "yolo_model_selection_charts/best_model_confusion_matrix.png", "Best Model Confusion Matrix"),
    ]
    for chart_path, title in charts:
        frame = image_page(chart_path, title)
        for _ in range(frames_per_page):
            out.write(frame)

    # --- Case study pages ---
    cases = [
        (IMG_BASE / "yolo_seg_eval_cases/tp_case_01.png", "TP Case - Defect Correctly Detected"),
        (IMG_BASE / "yolo_seg_eval_cases/fn_case_01.png", "FN Case - Missed Detection"),
        (IMG_BASE / "yolo_model_comparison_cases/both_TP_01.png", "Both Models TP - Large Defect"),
        (IMG_BASE / "yolo_model_comparison_cases/v22_better_01.png", "V2.2 Better Than V2.3"),
    ]
    for case_path, title in cases:
        frame = image_page(case_path, title)
        for _ in range(int(frames_per_page * 0.8)):
            out.write(frame)

    # --- Final page ---
    frame = text_page([
        "Conclusion",
        "",
        "YOLOv8n-seg (3 epochs) outperforms OpenCV CLAHE",
        "F1: 0.688 -> 0.833  (+21%)",
        "",
        "Full pipeline: Dataset -> OpenCV -> CLAHE -> YOLO -> Evaluation",
        "15 version tags  |  100+ experiments  |  Complete docs"
    ])
    for _ in range(frames_per_page):
        out.write(frame)

    out.release()
    print(f"\nVideo saved: {OUT_PATH.resolve()}")
    total_frames = sum([1 for _ in [1]])  # placeholder
    print(f"Duration: ~90 seconds (estimated)")

if __name__ == "__main__":
    generate()

