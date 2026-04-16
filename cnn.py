# ============================================================================
# GPU SETUP (ADD THIS)
# ============================================================================
print("\n🔍 Checking GPU...")

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Enable memory growth (prevents crashes)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print(f"✅ GPU detected: {len(gpus)} GPU(s)")
        print("Using:", gpus[0])

    except RuntimeError as e:
        print("❌ GPU setup error:", e)
else:
    print("⚠️ No GPU found. Running on CPU")
# ============================================================================
# IMPORTS
# ============================================================================
import os, zipfile, cv2, warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings('ignore')

print("TensorFlow:", tf.__version__)

# ============================================================================
# STEP 1: EXTRACT DATASET
# ============================================================================
def extract_dataset(zip_path='dataset.zip', extract_to='dataset'):
    if not os.path.exists(zip_path):
        print("❌ dataset.zip not found!")
        return False

    print("📦 Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print("✅ Extraction complete")
    return True


# ============================================================================
# PREPROCESS IMAGE
# ============================================================================
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)

        return img
    except:
        return None


# ============================================================================
# LOAD DATASET
# ============================================================================
def load_dataset(base_path='dataset'):
    X, y = [], []

    # Auto-detect path
    if os.path.exists(os.path.join(base_path, 'dataset')):
        base_path = os.path.join(base_path, 'dataset')

    print("📂 Using dataset path:", base_path)

    classes = {'debris': 0, 'non_debris': 1}

    for cls, label in classes.items():
        folder = os.path.join(base_path, cls)

        if not os.path.exists(folder):
            print(f"❌ Missing folder: {folder}")
            continue

        files = os.listdir(folder)
        print(f"📸 {cls}: {len(files)} images")

        for f in files:
            img = preprocess_image(os.path.join(folder, f))
            if img is not None:
                X.append(img)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"✅ Loaded {len(X)} images")
    return X, y


# ============================================================================
# MODEL
# ============================================================================
def build_model():
    model = Sequential([
        layers.Input(shape=(128,128,1)),

        Conv2D(32,3,activation='relu'), BatchNormalization(), MaxPooling2D(),
        Conv2D(64,3,activation='relu'), BatchNormalization(), MaxPooling2D(),
        Conv2D(128,3,activation='relu'), BatchNormalization(), MaxPooling2D(),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.6),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================================
# MAIN
# ============================================================================
def main():

    # STEP 1: Extract
    extract_dataset()

    # STEP 2: Load
    print("\n📥 Loading data...")
    X, y = load_dataset()

    if len(X) == 0:
        print("❌ No data loaded.")
        return

    # 🔥 Shuffle (IMPORTANT)
    X, y = shuffle(X, y, random_state=42)

    # 🔥 SPLIT: Train / Val / Test (70 / 15 / 15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print("\n📊 Dataset Split:")
    print("Train:", len(X_train))
    print("Validation:", len(X_val))
    print("Test:", len(X_test))

    # 🔥 DATA AUGMENTATION (only on training)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # STEP 3: Model
    model = build_model()
    model.summary()

    # 🔥 CALLBACKS (FIXED)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.3)
    ]

    # STEP 4: Train
    print("\n🚀 Training...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),   # ✅ CORRECT
        epochs=25,
        callbacks=callbacks
    )

    # STEP 5: Evaluate (ONLY ON TEST)
    print("\n📊 Evaluating on TEST set...")
    loss, acc = model.evaluate(X_test, y_test)
    print("✅ Test Accuracy:", acc)

    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    # 🔍 Duplicate Check
    print("\n🔍 Checking duplicate images...")
    unique_count = len(np.unique(X.reshape(len(X), -1), axis=0))
    print("Total images:", len(X))
    print("Unique images:", unique_count)

    if unique_count < len(X):
        print("⚠️ WARNING: Duplicate images detected!")

    # Save model
    model.save("debris_model.h5")
    print("\n✅ Model saved!")

# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    main()