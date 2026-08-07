"""
Military Aircraft Detection - Validation & Reporting
Author: Dhruv Gaur
GitHub: https://github.com/dhruvgaur10/Military-Aircraft-Detection

Evaluates a trained checkpoint on the held-out test split, copies the generated
charts into assets/ for the README, and writes a per-class metrics table in both
markdown and JSON.

Usage:
    python scripts/validate.py
    python scripts/validate.py --weights runs/detect/aircraft_v2/weights/best.pt
    python scripts/validate.py --split val --no-assets
"""

from ultralytics import YOLO
import argparse
import json
import os
import shutil

# Charts copied into assets/ for the README
CHART_FILES = [
    'confusion_matrix_normalized.png',
    'confusion_matrix.png',
    'results.png',
    'BoxPR_curve.png',
    'BoxF1_curve.png',
    'BoxP_curve.png',
    'BoxR_curve.png',
    'labels.jpg',
    'val_batch0_pred.jpg',
]


def resolve_weights(project_root, explicit):
    """Locate the checkpoint to evaluate."""
    if explicit:
        if not os.path.exists(explicit):
            raise SystemExit(f"Weights not found: {explicit}")
        return explicit

    # Prefer the newest best.pt under runs/detect, else fall back to models/best.pt
    runs_dir = os.path.join(project_root, 'runs', 'detect')
    candidates = []
    if os.path.isdir(runs_dir):
        for entry in os.listdir(runs_dir):
            w = os.path.join(runs_dir, entry, 'weights', 'best.pt')
            if os.path.exists(w):
                candidates.append((os.path.getmtime(w), w))

    if candidates:
        return max(candidates)[1]

    fallback = os.path.join(project_root, 'models', 'best.pt')
    if os.path.exists(fallback):
        return fallback

    raise SystemExit("No trained weights found. Train a model first.")


def collect_per_class(metrics, names):
    """Build a sorted per-class metrics table from the Ultralytics results object."""
    rows = []
    try:
        box = metrics.box
        # ap_class_index maps result rows back to class ids
        indices = list(box.ap_class_index)
        for i, cls_id in enumerate(indices):
            rows.append({
                'class': names[int(cls_id)],
                'precision': round(float(box.p[i]), 4),
                'recall': round(float(box.r[i]), 4),
                'mAP50': round(float(box.ap50[i]), 4),
                'mAP50_95': round(float(box.ap[i]), 4),
            })
    except (AttributeError, IndexError, TypeError) as e:
        print(f"  [!] Could not extract per-class metrics: {type(e).__name__}: {e}")
        return []

    rows.sort(key=lambda r: r['mAP50'], reverse=True)
    return rows


def write_markdown(rows, overall, out_path):
    """Write a README-ready markdown metrics table."""
    with open(out_path, 'w') as f:
        f.write("## Overall\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        for k, v in overall.items():
            f.write(f"| {k} | {v} |\n")

        if not rows:
            return

        f.write("\n## Per-Class Results\n\n")
        f.write("| Aircraft | Precision | Recall | mAP@50 | mAP@50-95 |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['class']} | {r['precision']:.3f} | {r['recall']:.3f} | "
                    f"{r['mAP50']:.3f} | {r['mAP50_95']:.3f} |\n")

        f.write("\n### Strongest 10\n\n")
        f.write("| Aircraft | mAP@50 |\n|---|---|\n")
        for r in rows[:10]:
            f.write(f"| {r['class']} | {r['mAP50']:.3f} |\n")

        f.write("\n### Weakest 10\n\n")
        f.write("| Aircraft | mAP@50 |\n|---|---|\n")
        for r in rows[-10:]:
            f.write(f"| {r['class']} | {r['mAP50']:.3f} |\n")


def main():
    parser = argparse.ArgumentParser(description='Validate a trained aircraft detector')
    parser.add_argument('--weights', type=str, default=None,
                        help='Checkpoint to evaluate (default: newest best.pt under runs/detect)')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'],
                        help='Dataset split to evaluate (default: test)')
    parser.add_argument('--imgsz', type=int, default=640, help='Input resolution (default: 640)')
    parser.add_argument('--batch', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--device', type=str, default=0, help='CUDA device or "cpu"')
    parser.add_argument('--no-assets', action='store_true',
                        help='Skip copying charts into assets/')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'data.yaml')
    weights = resolve_weights(project_root, args.weights)

    print("=" * 58)
    print("  MODEL VALIDATION")
    print("=" * 58)
    print(f"  Weights : {weights}")
    print(f"  Split   : {args.split}")
    print("=" * 58)

    model = YOLO(weights)
    metrics = model.val(
        data=data_path,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,
        save_json=True,
    )

    overall = {
        'mAP@50': f"{metrics.box.map50:.4f}",
        'mAP@50-95': f"{metrics.box.map:.4f}",
        'Precision': f"{metrics.box.mp:.4f}",
        'Recall': f"{metrics.box.mr:.4f}",
    }

    print("\n  OVERALL")
    for k, v in overall.items():
        print(f"    {k:<12}: {v}")

    names = model.names
    rows = collect_per_class(metrics, names)

    if rows:
        print(f"\n  Strongest classes:")
        for r in rows[:5]:
            print(f"    {r['class']:<14} mAP@50 = {r['mAP50']:.3f}")
        print(f"\n  Weakest classes:")
        for r in rows[-5:]:
            print(f"    {r['class']:<14} mAP@50 = {r['mAP50']:.3f}")

    # Persist reports next to the validation run
    save_dir = str(metrics.save_dir)
    json_path = os.path.join(save_dir, 'metrics_report.json')
    with open(json_path, 'w') as f:
        json.dump({
            'weights': weights,
            'split': args.split,
            'overall': overall,
            'per_class': rows,
        }, f, indent=4)

    md_path = os.path.join(save_dir, 'metrics_report.md')
    write_markdown(rows, overall, md_path)

    print(f"\n  JSON report : {json_path}")
    print(f"  MD report   : {md_path}")

    if not args.no_assets:
        assets_dir = os.path.join(project_root, 'assets')
        os.makedirs(assets_dir, exist_ok=True)
        copied = 0
        for fname in CHART_FILES:
            src = os.path.join(save_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(assets_dir, fname))
                copied += 1
        print(f"  Charts copied to assets/: {copied}")

    print("=" * 58)


if __name__ == "__main__":
    main()
