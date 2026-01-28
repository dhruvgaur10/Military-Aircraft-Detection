"""
Military Aircraft Detection - Detection Script
Author: [DHRUV GAUR]
GitHub: https://github.com/dhruvgaur10/Military-Aircraft-Detection
"""

from ultralytics import YOLO
import argparse
import os

def detect(source, model_path=None, confidence=0.25):
    """
    Detect military aircraft in image/video
    
    Args:
        source: Path to image, video, or camera index (0)
        model_path: Path to model weights
        confidence: Detection confidence threshold
    """
    print("=" * 50)
    print("MILITARY AIRCRAFT DETECTION")
    print("=" * 50)
    
    # Get default model path
    if model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_path = os.path.join(project_root, 'models', 'best.pt')
    
    print(f"Model: {model_path}")
    print(f"Source: {source}")
    print(f"Confidence: {confidence}")
    print("=" * 50)
    
    # Load model
    model = YOLO(model_path)
    
    # Run detection
    results = model.predict(
        source=source,
        conf=confidence,
        save=True,
        show=True
    )
    
    print("=" * 50)
    print("✅ DETECTION COMPLETE!")
    print(f"Results saved to: {results[0].save_dir}")
    print("=" * 50)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Military Aircraft Detection')
    parser.add_argument('--source', type=str, required=True, 
                        help='Image/video path or camera index (0 for webcam)')
    parser.add_argument('--model', type=str, default=None, 
                        help='Path to model weights (default: models/best.pt)')
    parser.add_argument('--conf', type=float, default=0.25, 
                        help='Confidence threshold (default: 0.25)')
    
    args = parser.parse_args()
    detect(args.source, args.model, args.conf)