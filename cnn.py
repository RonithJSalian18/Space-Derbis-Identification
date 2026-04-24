# ============================================================================
# IMPORTS
# ============================================================================
import tensorflow as tf
import os, zipfile, cv2, warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.utils import shuffle

from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings('ignore')

print("TensorFlow:", tf.__version__)

# ============================================================================
# GPU CHECK
# ============================================================================
print("\n🔍 Checking GPU...")
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU detected: {len(gpus)}")
else:
    print("⚠️ No GPU found. Running on CPU")

# ============================================================================
# EXTRACT DATASET
# ============================================================================
def extract_dataset(zip_path='dataset.zip', extract_to='dataset'):
    if os.path.exists(zip_path):
        print("📦 Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("✅ Done")

# ============================================================================
# PREPROCESS
# ============================================================================
def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (128,128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    return np.expand_dims(img, axis=-1)

# ============================================================================
# LOAD DATASET (🔥 FIXED AUTO DETECT)
# ============================================================================
def load_dataset(base_path='dataset'):
    X, y = [], []

    print("🔍 Searching dataset...")

    # Auto-detect correct folder
    data_path = None

    for root, dirs, _ in os.walk(base_path):
        if 'debris' in dirs and 'non_debris' in dirs:
            data_path = root
            break

    if data_path is None:
        print("❌ Dataset not found!")
        return np.array([]), np.array([])

    print("✅ Using:", data_path)

    classes = {'debris':0, 'non_debris':1}

    for cls, label in classes.items():
        folder = os.path.join(data_path, cls)

        files = os.listdir(folder)
        print(f"{cls}: {len(files)} images")

        for f in files:
            img = preprocess_image(os.path.join(folder,f))
            if img is not None:
                X.append(img)
                y.append(label)

    return np.array(X), np.array(y)

# ============================================================================
# REMOVE DUPLICATES (🔥 IMPORTANT)
# ============================================================================
def remove_duplicates(X, y):
    print("\n🧹 Removing duplicates...")

    seen = set()
    X_clean, y_clean = [], []

    for img, label in zip(X, y):
        h = hash(img.tobytes())
        if h not in seen:
            seen.add(h)
            X_clean.append(img)
            y_clean.append(label)

    print(f"Before: {len(X)} | After: {len(X_clean)}")

    return np.array(X_clean), np.array(y_clean)

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
        Dropout(0.5),

        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
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
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.title("ROC Curve")
    plt.show()

def plot_pr(y_test, y_prob):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision)
    plt.title("Precision-Recall")
    plt.show()

# ============================================================================
# MAIN
# ============================================================================
def main():

    extract_dataset()

    print("\n📥 Loading...")
    X, y = load_dataset()

    if len(X) == 0:
        return

    # Shuffle
    X, y = shuffle(X, y, random_state=42)

    # 🔥 REMOVE DUPLICATES
    X, y = remove_duplicates(X, y)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    model = build_model()

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=25,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(patience=3)
        ]
    )

    plot_training(history)

    loss, acc = model.evaluate(X_test, y_test)
    print("Test Accuracy:", acc)

    # 🔥 FIXED PREDICTION
    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int)

    print(classification_report(y_test, y_pred))

    plot_confusion(y_test, y_pred)
    plot_roc(y_test, y_prob)
    plot_pr(y_test, y_prob)

    model.save("debris_model_final.h5")
    print("✅ Model saved")

# ============================================================================
if __name__ == "__main__":
    main()