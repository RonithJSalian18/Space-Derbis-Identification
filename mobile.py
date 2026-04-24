# ============================================================================
# IMPORTS
# ============================================================================
import os, zipfile, cv2, warnings, hashlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
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
# GPU ENABLE
# ============================================================================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ============================================================================
# EXTRACT DATASET
# ============================================================================
def extract_dataset(zip_path='dataset.zip', extract_to='dataset'):
    if os.path.exists(zip_path) and not os.path.exists(extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("✅ Dataset extracted")

# ============================================================================
# PREPROCESS
# ============================================================================
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)

    return img

# ============================================================================
# FIND DATASET PATH (FIXED)
# ============================================================================
def find_dataset_path(base='dataset'):
    for root, dirs, _ in os.walk(base):
        if 'debris' in dirs and 'non_debris' in dirs:
            return root
    raise Exception("Dataset folders not found")

# ============================================================================
# LOAD DATA
# ============================================================================
def load_dataset():
    base_path = find_dataset_path()

    X, y = [], []
    classes = {'debris':0, 'non_debris':1}

    for cls, label in classes.items():
        folder = os.path.join(base_path, cls)

        for f in os.listdir(folder):
            img = preprocess_image(os.path.join(folder, f))
            if img is not None:
                X.append(img)
                y.append(label)

    return np.array(X), np.array(y)

# ============================================================================
# REMOVE DUPLICATES
# ============================================================================
def remove_duplicates(X, y):
    seen = set()
    X_clean, y_clean = [], []

    for img, label in zip(X, y):
        h = hashlib.md5(img.tobytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            X_clean.append(img)
            y_clean.append(label)

    print(f"Before: {len(X)} After: {len(X_clean)}")
    return np.array(X_clean), np.array(y_clean)

# ============================================================================
# MODEL
# ============================================================================
def build_model():
    base_model = MobileNetV2(
        input_shape=(128, 128, 3),
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
# PLOTS
# ============================================================================
def plot_training(history):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title("Accuracy vs Epoch")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title("Loss vs Epoch")
    plt.legend()

    plt.show()


def plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['Debris','Non-Debris'],
                yticklabels=['Debris','Non-Debris'])
    plt.title("Confusion Matrix")
    plt.show()


def plot_roc(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


def plot_pr(y_test, y_prob):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.show()

# ============================================================================
# MAIN
# ============================================================================
def main():

    extract_dataset()

    print("\n📥 Loading data...")
    X, y = load_dataset()

    if len(X) == 0:
        print("❌ No data")
        return

    # 🔥 REMOVE DUPLICATES
    X, y = remove_duplicates(X, y)

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
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=20,
        callbacks=callbacks
    )

    # 📊 TRAINING GRAPHS
    plot_training(history)

    print("\n📊 Testing...")
    loss, acc = model.evaluate(X_test, y_test)
    print("Test Accuracy:", acc)

    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int)

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    # 📊 EVALUATION GRAPHS
    plot_confusion(y_test, y_pred)
    plot_roc(y_test, y_prob)
    plot_pr(y_test, y_prob)

    model.save("mobilenet_debris.h5")
    print("✅ Model saved!")

# ============================================================================
if __name__ == "__main__":
    main()