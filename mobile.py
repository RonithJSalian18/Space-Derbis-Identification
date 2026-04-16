# ============================================================================
# IMPORTS
# ============================================================================
import os, zipfile, cv2, warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings('ignore')

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# ============================================================================
# GPU ENABLE (IMPORTANT)
# ============================================================================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ============================================================================
# EXTRACT DATASET
# ============================================================================
def extract_dataset(zip_path='dataset.zip', extract_to='dataset'):
    if not os.path.exists(zip_path):
        print("❌ dataset.zip not found!")
        return False

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print("✅ Dataset extracted")
    return True

# ============================================================================
# PREPROCESS IMAGE (FIXED FOR MOBILENET)
# ============================================================================
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        img = cv2.resize(img, (128, 128))

        # ✅ KEEP RGB (NO GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ✅ MobileNet preprocessing
        img = preprocess_input(img)

        return img
    except:
        return None

# ============================================================================
# LOAD DATASET
# ============================================================================
def load_dataset(base_path='dataset'):
    X, y = [], []

    if os.path.exists(os.path.join(base_path, 'dataset')):
        base_path = os.path.join(base_path, 'dataset')

    print("📂 Dataset path:", base_path)

    classes = {'debris': 0, 'non_debris': 1}

    for cls, label in classes.items():
        folder = os.path.join(base_path, cls)

        if not os.path.exists(folder):
            continue

        files = os.listdir(folder)
        print(f"{cls}: {len(files)}")

        for f in files:
            img = preprocess_image(os.path.join(folder, f))
            if img is not None:
                X.append(img)
                y.append(label)

    return np.array(X), np.array(y)

# ============================================================================
# BUILD MODEL (FIXED)
# ============================================================================
def build_model():
    base_model = MobileNetV2(
        input_shape=(128, 128, 3),   # ✅ FIXED
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# ============================================================================
# MAIN
# ============================================================================
def main():

    extract_dataset()

    print("\nLoading data...")
    X, y = load_dataset()

    if len(X) == 0:
        print("❌ No data")
        return

    # SPLIT
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print("Train:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))

    # AUGMENTATION
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=3)
    ]

    print("\n🚀 Training...")
    model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=20,
        callbacks=callbacks
    )

    print("\n📊 Testing...")
    loss, acc = model.evaluate(X_test, y_test)
    print("Test Accuracy:", acc)

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(classification_report(y_test, y_pred))

    model.save("mobilenet_debris.h5")
    print("✅ Model saved!")

# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    main()