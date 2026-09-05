import os
import sys
import cv2
import numpy as np
import tensorflow as tf

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data.preprocessing import preprocess_image
from src.models import ModelFactory


class DebrisPredictor:
    """
    Inference class for loading a trained model and classifying space images.
    Supports both saved model files and checkpoint weight files.
    """
    def __init__(self, model_path, model_type="cnn"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

        print(f"📦 Loading model / weights from {model_path}...")
        self.model_type = model_type.lower()
        self.model, self.color_mode = ModelFactory.create_model(architecture_name=self.model_type)

        try:
            # Unfreeze top backbone layers if transfer learning architecture
            if "resnet" in self.model_type:
                from src.models import unfreeze_resnet
                unfreeze_resnet(self.model)
            elif "efficientnet" in self.model_type or "effinet" in self.model_type:
                from src.models import unfreeze_efficientnet
                unfreeze_efficientnet(self.model)
            elif "mobilenet" in self.model_type:
                from src.models import unfreeze_mobilenet
                unfreeze_mobilenet(self.model)


            # Try loading weights first if weights file
            self.model.load_weights(model_path)
            print("✅ Model weights loaded successfully!")
        except Exception as err:
            # Fallback to full model load
            try:
                self.model = tf.keras.models.load_model(model_path)
                print("✅ Full model loaded successfully!")
            except Exception as e:
                raise ValueError(f"Could not load model weights or full model from {model_path}: {err} | {e}")

    def predict(
        self,
        image_path: str,
        threshold: float = 0.5,
        allow_uncertain: bool = True,
        uncertain_lower: float = 0.35,
        uncertain_upper: float = 0.65
    ) -> dict:
        """
        Classify a single image file path.

        Args:
            image_path (str): Target image filepath.
            threshold (float): Calibrated decision threshold.
            allow_uncertain (bool): If True, flags predictions inside [uncertain_lower, uncertain_upper] as 'Uncertain'.
            uncertain_lower (float): Lower probability bound for uncertainty zone.
            uncertain_upper (float): Upper probability bound for uncertainty zone.

        Returns:
            dict: Structured prediction dictionary with label, confidence, status, and calibrated probabilities.
        """
        img_tensor = preprocess_image(image_path, color_mode=self.color_mode, model_type=self.model_type)
        if img_tensor is None:
            return {"error": f"Could not read image from {image_path}"}

        batch_tensor = np.expand_dims(img_tensor, axis=0)

        # Raw output (sigmoid probability of class 1: Non-Debris)
        prob_non_debris = float(self.model.predict(batch_tensor, verbose=0)[0][0])
        prob_debris = 1.0 - prob_non_debris

        if allow_uncertain and (uncertain_lower <= prob_non_debris <= uncertain_upper):
            label = "Uncertain"
            status = "UNCERTAIN (FLAGGED FOR HUMAN OR MULTI-SENSOR AUDIT)"
            confidence = max(prob_debris, prob_non_debris)
        elif prob_non_debris > threshold:
            label = "Non-Debris"
            status = "CONFIDENT NON-DEBRIS SPACECRAFT"
            confidence = prob_non_debris
        else:
            label = "Debris"
            status = "CONFIDENT SPACE DEBRIS"
            confidence = prob_debris

        return {
            "image_path": image_path,
            "prediction": label,
            "status": status,
            "confidence": round(confidence * 100, 2),
            "prob_debris": round(prob_debris, 4),
            "prob_non_debris": round(prob_non_debris, 4),
            "threshold_applied": threshold
        }
