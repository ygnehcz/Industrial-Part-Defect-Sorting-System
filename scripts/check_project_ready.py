"""Project Readiness Check脚本"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def check_import(name, pkg=None):
    try:
        __import__(pkg or name)
        return True, "OK"
    except ImportError:
        return False, "NOT FOUND"

def check_file(path):
    return Path(path).exists()

def main():
    print("=" * 50)
    print("Project Readiness Check")
    print("=" * 50)
    all_ok = True

    # Python deps
    print("\n[Python Deps]")
    deps = [("opencv-python", "cv2"), ("numpy", "numpy"), ("pandas", "pandas"),
            ("matplotlib", "matplotlib"), ("ultralytics", "ultralytics")]
    for name, pkg in deps:
        ok, msg = check_import(name, pkg)
        tag = "[OK]" if ok else "[MISSING]"
        if not ok: all_ok = False
        note = "" if ok else (" (YOLO demo 不可用)" if name=="ultralytics" else "")
        print(f"  {tag} {name}{note}")

    # Key files
    print("\n[Key Files]")
    files = ["README.md", "config.py", "main.py", "demo_run.py",
             "data/yolo_seg/data.yaml",
             "outputs/yolo_seg/runs/yolov8n_seg_smoke/weights/best.pt"]
    for f in files:
        ok = check_file(f)
        tag = "[OK]" if ok else "[MISSING]"
        if not ok: all_ok = False
        print(f"  {tag} {f}")

    # Key docs
    print("\n[Key Docs]")
    docs = ["docs/final_project_report.md", "docs/interview_questions.md",
            "docs/resume_snippets.md", "docs/yolo_model_selection_analysis.md"]
    for d in docs:
        ok = check_file(d)
        tag = "[OK]" if ok else "[MISSING]"
        if not ok: all_ok = False
        print(f"  {tag} {d}")

    print(f"\n{'='*50}")
    if all_ok:
        print("Project Ready：All deps, files, and docs in place.。")
    else:
        print("Project NOT Ready：Fix the above [MISSING] items。")
    print("=" * 50)


if __name__ == "__main__":
    main()

