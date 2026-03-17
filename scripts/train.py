"""
Military Aircraft Detection - Training Script
Author: Dhruv Gaur
GitHub: https://github.com/dhruvgaur10/Military-Aircraft-Detection
"""

from ultralytics import YOLO
import os


def train():
    print("=" * 50)
    print("MILITARY AIRCRAFT DETECTION - TRAINING")
    print("=" * 50)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    data_path = os.path.join(project_root, 'data', 'data.yaml')
    print(f"Data path: {data_path}")

    model = YOLO('yolov8n.pt')

    model.train(
        data=data_path,
        epochs=50,
        imgsz=640,
        batch=16,
        name='military_aircraft',
        patience=20,
        save=True,
        plots=True
    )

    print("=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    train()
