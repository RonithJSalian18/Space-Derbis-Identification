"""
Unified Command Line Inference Script for Space Debris Identification.

Usage examples:
    python predict.py --image path/to/image.jpg --model saved_models/cnn_debris.h5 --type cnn
"""
import argparse
import os
from src.utils import setup_gpu
from src.inference import DebrisPredictor


def main():
    parser = argparse.ArgumentParser(description="Run Space Debris Identification Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--model", type=str, default="saved_models/cnn_debris.h5", help="Path to trained .h5 model file")
    parser.add_argument("--type", type=str, default="cnn", choices=["cnn", "mobilenet", "resnet", "efficientnet"],
                        help="Model type (cnn or transfer learning model)")

    args = parser.parse_args()

    setup_gpu()

    predictor = DebrisPredictor(model_path=args.model, model_type=args.type)
    res = predictor.predict(image_path=args.image)

    print("\n==================================================")
    print("📸 INFERENCE RESULT")
    print("==================================================")
    print(f"File Path:       {res.get('image_path')}")
    print(f"Prediction:      {res.get('prediction')}")
    print(f"Confidence:      {res.get('confidence')}%")
    print(f"Debris Prob:     {res.get('prob_debris')}")
    print(f"Non-Debris Prob: {res.get('prob_non_debris')}")
    print("==================================================")


if __name__ == "__main__":
    main()
