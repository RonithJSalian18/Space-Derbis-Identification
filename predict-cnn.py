import cv2
import numpy as np
import tensorflow as tf

# ============================================================================
# LOAD MODEL
# ============================================================================
model = tf.keras.models.load_model("D:\Space-Debris-vs\Space-models\mobile\mobilenet_debris(98).h5")
print("✅ Model loaded!")

# ============================================================================
# PREPROCESS IMAGE (MUST MATCH TRAINING)
# ============================================================================
def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Could not read image")
        return None

    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)

    return img


# ============================================================================
# PREDICT FUNCTION
# ============================================================================
def predict_image(image_path):
    img = preprocess_image(image_path)

    if img is None:
        return

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Prediction
    prob = model.predict(img)[0][0]

    # Convert to label
    if prob > 0.5:
        label = "Non-Debris"
        confidence = prob
    else:
        label = "Debris"
        confidence = 1 - prob

    print("\n📸 Image:", image_path)
    print("🔍 Prediction:", label)
    print(f"📊 Confidence: {confidence*100:.2f}%")
    print(f"Debris Prob: {1-prob:.4f}")
    print(f"Non-Debris Prob: {prob:.4f}")


# ============================================================================
# TEST
# ============================================================================
predict_image(r"D:\Space Debris\Datasets\new_set\non_debris\img000010_jpg.rf.pVIDKGHsoTqlv3syeTzE.jpg")   # 👉 put your image path here