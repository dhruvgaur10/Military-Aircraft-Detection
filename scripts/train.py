"""
Military Aircraft Detection - Training Script
Author: Dhruv Gaur
GitHub: https://github.com/dhruvgaur10/Military-Aircraft-Detection

Training configuration tuned for maximum mAP on a 6GB GPU.

The previous 50-epoch YOLOv8n run reached 0.272 mAP50 with both mAP curves still
climbing at cutoff, so the schedule here is much longer and the default backbone
is larger. Key changes over that run:

  - 300 epochs instead of 50   (the earlier run stopped mid-climb)
  - YOLOv8s instead of YOLOv8n (nano lacks the capacity to separate similar planforms)
  - cosine LR schedule         (better convergence over a long run)
  - stronger augmentation      (mixup + copy-paste target the false-negative rate)
  - AdamW with a lower LR      (more stable than SGD for fine-grained classes)

Usage:
    python scripts/train.py
    python scripts/train.py --model yolov8m.pt --batch 6
    python scripts/train.py --epochs 150 --name quick_run
    python scripts/train.py --resume
"""

from ultralytics import YOLO
import argparse
import os


def build_config(args, data_path):
    """Assemble the Ultralytics training keyword arguments."""
    return dict(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        workers=args.workers,

        # --- Schedule -------------------------------------------------------
        # Long runs benefit from cosine decay; patience is generous because the
        # previous run proved this dataset keeps improving late.
        optimizer='AdamW',
        lr0=0.002,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5.0,
        weight_decay=0.0005,
        momentum=0.937,
        patience=75,

        # --- Loss weighting -------------------------------------------------
        # cls is raised from the 0.5 default: with 40+ visually similar classes
        # the classification term is the hard part, not localisation.
        box=7.5,
        cls=1.0,
        dfl=1.5,

        # --- Augmentation ---------------------------------------------------
        # mixup and copy_paste add object density, which directly attacks the
        # background false-negative pattern seen in the previous confusion matrix.
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.10,
        copy_paste=0.10,
        erasing=0.4,
        close_mosaic=15,

        # --- Runtime --------------------------------------------------------
        amp=True,
        cache=args.cache,
        save=True,
        save_period=25,
        plots=True,
        val=True,
        seed=0,
        deterministic=True,
        exist_ok=True,
        resume=args.resume,
    )


def train():
    parser = argparse.ArgumentParser(description='Train the aircraft detector')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        help='Base weights (default: yolov8s.pt)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Training epochs (default: 300)')
    parser.add_argument('--batch', type=int, default=12,
                        help='Batch size, tuned for 6GB VRAM (default: 12)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input resolution (default: 640)')
    parser.add_argument('--name', type=str, default='aircraft_v2',
                        help='Run name under runs/detect (default: aircraft_v2)')
    parser.add_argument('--device', type=str, default=0,
                        help='CUDA device index or "cpu" (default: 0)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Dataloader workers (default: 8)')
    parser.add_argument('--cache', type=str, default='disk',
                        help='Image cache: disk, ram, or None (default: disk)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume the last interrupted run')
    args = parser.parse_args()

    if args.cache in ('None', 'none', ''):
        args.cache = False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'data.yaml')

    if not os.path.exists(data_path):
        raise SystemExit(
            f"data.yaml not found at {data_path}\n"
            "Run scripts/prepare_dataset.py first to build the dataset."
        )

    print("=" * 58)
    print("  MILITARY AIRCRAFT DETECTION - TRAINING")
    print("=" * 58)
    print(f"  Model      : {args.model}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch      : {args.batch}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Data       : {data_path}")
    print("=" * 58)

    model = YOLO(args.model)
    results = model.train(**build_config(args, data_path))

    print("=" * 58)
    print("  TRAINING COMPLETE")
    print(f"  Results saved to: {results.save_dir}")
    print("  Next: python scripts/validate.py --weights <run>/weights/best.pt")
    print("=" * 58)


if __name__ == "__main__":
    train()
