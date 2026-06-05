# Demo Video Script

## Video

- Path: `outputs/demo/project_demo_v3.mp4`
- Resolution: 1280x720, 24fps
- Duration: ~71 seconds
- Size: ~7MB

## Structure

| # | Page | Content | Duration |
|---|------|---------|----------|
| 1 | Title | Project name + subtitle | 5s |
| 2 | Roadmap | V1→V2→V3 versions | 5s |
| 3 | Dataset | 3335 pairs, OK/NG counts | 5s |
| 4 | OpenCV CLAHE | F1=0.688, TP/TN/FP/FN | 5s |
| 5 | YOLO Best | F1=0.833, conf=0.05 | 5s |
| 6 | Recall Priority | F1=0.832, conf=0.03 | 5s |
| 7 | V1.1 vs V1.2 chart | Comparison bar chart | 5s |
| 8 | F1 comparison chart | All versions F1 | 5s |
| 9 | PR curve chart | Precision-Recall tradeoff | 5s |
| 10 | Confusion matrix | Best model matrix | 5s |
| 11 | TP case | Correct detection example | 4s |
| 12 | FN case | Missed detection example | 4s |
| 13 | Both TP case | Model comparison example | 4s |
| 14 | V2.2 better case | 3ep vs 10ep | 4s |
| 15 | Conclusion | Summary + GitHub | 5s |

## How to Present with This Video

1. Start video, let it play through once
2. Pause on the YOLO Best page to explain the metrics
3. Pause on the PR curve to explain Precision-Recall tradeoff
4. Pause on the FN case to explain real-world limitations
5. End with "Code and full docs on GitHub"

## Recommended Narration Script

"This is an industrial surface defect detection system I built. It uses 3,335 real industrial images with pixel-level GT masks. I started with traditional OpenCV methods - the baseline fixed threshold had only 4.5% recall. After adding CLAHE contrast enhancement, F1 reached 0.688. Then I converted the GT masks to YOLO segmentation format and trained YOLOv8n-seg. After just 3 epochs, F1 reached 0.833, with 90.4% precision and 77.3% recall. The project includes over 100 parameter experiments, full IoU/Dice segmentation evaluation, and model selection analysis. All code and documentation are on GitHub."

## Charts Used

- `docs/images/experiment_charts/v1_1_vs_v1_2_metrics.png`
- `docs/images/yolo_training_charts/yolo_f1_comparison.png`
- `docs/images/yolo_model_selection_charts/model_conf_precision_recall_curve.png`
- `docs/images/yolo_model_selection_charts/best_model_confusion_matrix.png`

## Case Images Used

- `docs/images/yolo_seg_eval_cases/tp_case_01.png`
- `docs/images/yolo_seg_eval_cases/fn_case_01.png`
- `docs/images/yolo_model_comparison_cases/both_TP_01.png`
- `docs/images/yolo_model_comparison_cases/v22_better_01.png`
