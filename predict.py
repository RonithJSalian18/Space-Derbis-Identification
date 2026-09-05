"""
Unified Command Line Inference Script for Space Debris Identification.

Usage examples:
    python predict.py --image path/to/image.jpg --model saved_models/cnn_debris.h5 --type cnn
"""
import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.utils import setup_gpu
from src.inference import DebrisPredictor


def main():
    parser = argparse.ArgumentParser(description="Run Space Debris Identification Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--model", type=str, default="saved_models/cnn_spark_debris.h5", help="Path to trained .h5 model file")
    parser.add_argument("--type", type=str, default="cnn", choices=["cnn", "mobilenet", "resnet", "efficientnet"],
                        help="Model type (cnn or transfer learning model)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for debris vs non-debris")

    args = parser.parse_args()

    setup_gpu()

    predictor = DebrisPredictor(model_path=args.model, model_type=args.type)
    res = predictor.predict(image_path=args.image, threshold=args.threshold)

    print("\n==================================================")
    print("[+] INFERENCE RESULT (SpaceGuard Vision Engine)")
    print("==================================================")
    print(f"File Path:        {res.get('image_path')}")
    print(f"Prediction:       {res.get('prediction')}")
    print(f"Status:           {res.get('status', 'N/A')}")
    print(f"Confidence:       {res.get('confidence')}%")
    print(f"P(Debris):        {res.get('prob_debris')}")
    print(f"P(Non-Debris):    {res.get('prob_non_debris')}")
    print(f"Threshold:        {res.get('threshold_applied', args.threshold)}")
    print("==================================================")


if __name__ == "__main__":
    main()
