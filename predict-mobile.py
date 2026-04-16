# ============================================================================
# IMPORTS
# ============================================================================
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================================
# LOAD MODEL
# ============================================================================
model = load_model("D:\Space-Debris-vs\Space-models\mobile\mobilenet_debris(98).h5")
print("✅ Model loaded!")

# ============================================================================
# PREPROCESS FUNCTION (MUST MATCH TRAINING)
# ============================================================================
def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Error: Image not found or path is wrong")
        return None

    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # MobileNet preprocessing
    img = preprocess_input(img)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

# ============================================================================
# PREDICT FUNCTION
# ============================================================================
def predict_image(image_path):
    img = preprocess_image(image_path)

    if img is None:
        return

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        label = "Non-Debris"
        confidence = prediction
    else:
        label = "Debris"
        confidence = 1 - prediction

    print("\n📊 Prediction Result:")
    print("Image:", image_path)
    print("Prediction:", label)
    print("Confidence:", round(float(confidence) * 100, 2), "%")

# ============================================================================
# TEST
# ============================================================================
# 👉 CHANGE THIS PATH
predict_image(r"D:\Space Debris\Datasets\new_set\non_debris\img000029_jpg.rf.cbsZu3TPpQX699Mva69R.jpg")