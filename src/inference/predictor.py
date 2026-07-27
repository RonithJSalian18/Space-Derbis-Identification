import os
import cv2
import numpy as np
import tensorflow as tf
from src.data.preprocessing import preprocess_image


class DebrisPredictor:
    """
    Inference class for loading a trained model and classifying space images.
    """
    def __init__(self, model_path, model_type="cnn"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

        print(f"📦 Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        self.model_type = model_type.lower()
        self.color_mode = "grayscale" if self.model_type == "cnn" else "rgb"
        print("✅ Model loaded successfully!")

    def predict(self, image_path, threshold=0.5):
        """
        Classify a single image file path.
        Returns dictionary with prediction label, confidence, and raw probabilities.
        """
        img_tensor = preprocess_image(image_path, color_mode=self.color_mode)
        if img_tensor is None:
            return {"error": f"Could not read image from {image_path}"}

        # Add batch dimension
        batch_tensor = np.expand_dims(img_tensor, axis=0)

        # Raw output (sigmoid probability of class 1: Non-Debris)
        prob_non_debris = float(self.model.predict(batch_tensor, verbose=0)[0][0])
        prob_debris = 1.0 - prob_non_debris

        if prob_non_debris > threshold:
            label = "Non-Debris"
            confidence = prob_non_debris
        else:
            label = "Debris"
            confidence = prob_debris

        return {
            "image_path": image_path,
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "prob_debris": round(prob_debris, 4),
            "prob_non_debris": round(prob_non_debris, 4)
        }
