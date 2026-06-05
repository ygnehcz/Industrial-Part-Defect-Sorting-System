import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from config import (
    SAMPLE_IMAGE_DIR,
    IMAGE_OUTPUT_DIR,
    REPORT_DIR,
    THRESHOLD,
    THRESHOLD_LIST,
    THRESHOLD_EXPERIMENT_CSV,
    THRESHOLD_METRICS_FIGURE,
    MORPH_OUTPUT_DIR
)
from src.batch_detector import (
    run_batch_detection,
    run_threshold_experiments,
    find_best_threshold
)
from src.evaluator import evaluate_results
from src.report import save_report, save_metrics, save_threshold_experiments
from src.visualizer import load_threshold_experiment, plot_threshold_metrics


class DefectDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("工业零件表面缺陷检测系统")
        self.root.geometry("600x500")

        self.image_dir = tk.StringVar(value=SAMPLE_IMAGE_DIR)
        self.status_text = tk.StringVar(value="就绪，请选择图片文件夹后点击开始检测")

        self._build_ui()

    def _build_ui(self):
        # 文件夹选择
        frame_dir = tk.LabelFrame(self.root, text="图片文件夹", padx=10, pady=10)
        frame_dir.pack(fill="x", padx=10, pady=5)

        tk.Entry(frame_dir, textvariable=self.image_dir, width=60).pack(side="left", padx=(0, 5))
        tk.Button(frame_dir, text="浏览...", command=self._browse_folder).pack(side="left")

        # 检测按钮
        frame_action = tk.Frame(self.root)
        frame_action.pack(fill="x", padx=10, pady=5)

        tk.Button(
            frame_action, text="开始检测", command=self._run_detection,
            bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
            width=15, height=2
        ).pack()

        # 结果显示
        frame_result = tk.LabelFrame(self.root, text="检测结果", padx=10, pady=10)
        frame_result.pack(fill="both", expand=True, padx=10, pady=5)

        self.result_text = tk.Text(frame_result, height=14, width=70, state="disabled")
        self.result_text.pack(fill="both", expand=True)

        # 状态栏
        frame_status = tk.Frame(self.root)
        frame_status.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_status, textvariable=self.status_text, fg="gray").pack(side="left")

    def _browse_folder(self):
        folder = filedialog.askdirectory(initialdir=SAMPLE_IMAGE_DIR)
        if folder:
            self.image_dir.set(folder)

    def _set_result(self, text):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)
        self.result_text.config(state="disabled")

    def _run_detection(self):
        self.status_text.set("检测中，请稍候...")
        self.root.update()

        try:
            sample_dir = Path(self.image_dir.get())
            image_output_dir = Path(IMAGE_OUTPUT_DIR)
            report_dir = Path(REPORT_DIR)
            morph_output_dir = Path(MORPH_OUTPUT_DIR)

            image_output_dir.mkdir(parents=True, exist_ok=True)
            report_dir.mkdir(parents=True, exist_ok=True)
            morph_output_dir.mkdir(parents=True, exist_ok=True)

            image_paths = sorted(sample_dir.glob("*.jpg"))
            if not image_paths:
                messagebox.showwarning("提示", "所选文件夹中没有找到 .jpg 图片")
                self.status_text.set("未找到图片")
                return

            results = run_batch_detection(image_paths, image_output_dir, morph_output_dir, THRESHOLD)
            metrics = evaluate_results(results)

            detail_report_path = report_dir / "batch_detection_report.csv"
            summary_report_path = report_dir / "batch_summary_metrics.csv"

            save_report(results, detail_report_path)
            save_metrics(metrics, summary_report_path)

            experiment_results = run_threshold_experiments(image_paths, image_output_dir, morph_output_dir)

            threshold_report_path = Path(THRESHOLD_EXPERIMENT_CSV)
            save_threshold_experiments(experiment_results, threshold_report_path)

            loaded_experiments = load_threshold_experiment(threshold_report_path)
            figure_path = Path(THRESHOLD_METRICS_FIGURE)
            plot_threshold_metrics(loaded_experiments, figure_path)

            best_result = find_best_threshold(experiment_results)

            total = len(image_paths)
            ok_count = sum(1 for r in results if r["pred_status"] == "OK")
            ng_count = sum(1 for r in results if r["pred_status"] == "NG")

            result_text = (
                f"检测总数：{total}\n"
                f"OK 数量：{ok_count}\n"
                f"NG 数量：{ng_count}\n"
                f"Accuracy 准确率：{metrics['accuracy'] * 100:.1f}%\n"
                f"Precision 精确率：{metrics['precision'] * 100:.1f}%\n"
                f"Recall 召回率：{metrics['recall'] * 100:.1f}%\n"
                f"推荐阈值：{best_result['threshold']}\n"
                f"\n"
                f"检测明细报表：{detail_report_path}\n"
                f"汇总评价报表：{summary_report_path}\n"
                f"阈值实验报表：{threshold_report_path}\n"
                f"阈值实验曲线图：{figure_path}\n"
            )

            self._set_result(result_text)
            self.status_text.set("检测完成")

        except Exception as e:
            messagebox.showerror("错误", f"检测过程出错：\n{str(e)}")
            self.status_text.set("检测失败")


def main():
    root = tk.Tk()
    app = DefectDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
