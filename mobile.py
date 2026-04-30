# ============================================================================
# IMPORTS
# ============================================================================
import os, zipfile, cv2, warnings, random, hashlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
import imagehash

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

warnings.filterwarnings('ignore')

# ============================================================================
# SEED
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================================
# DATASET UTILITIES
# ============================================================================
def extract_dataset(zip_path='dataset.zip', extract_to='dataset'):
    if os.path.exists(zip_path) and not os.path.exists(extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

def find_dataset_path(base='dataset'):
    for root, dirs, _ in os.walk(base):
        if 'debris' in dirs and 'non_debris' in dirs:
            return root
    raise Exception("Dataset folders not found")

# ============================================================================
# CLEAN INVALID IMAGES
# ============================================================================
def clean_images(folder):
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        img = cv2.imread(path)
        if img is None or img.shape[0] < 50 or img.shape[1] < 50:
            os.remove(path)

# ============================================================================
# LOAD PATHS
# ============================================================================
def load_paths():
    base = find_dataset_path()
    paths, labels = [], []

    for cls, label in {'debris':0, 'non_debris':1}.items():
        folder = os.path.join(base, cls)
        for f in os.listdir(folder):
            paths.append(os.path.join(folder, f))
            labels.append(label)

    return np.array(paths), np.array(labels)

# ============================================================================
# REMOVE EXACT DUPLICATES
# ============================================================================
def remove_exact_duplicates(paths, labels):
    seen = {}
    new_paths, new_labels = [], []

    for p, l in zip(paths, labels):
        img = cv2.imread(p)
        if img is None:
            continue
        h = hashlib.md5(img.tobytes()).hexdigest()

        if h not in seen:
            seen[h] = True
            new_paths.append(p)
            new_labels.append(l)

    print("Exact duplicates removed:", len(paths) - len(new_paths))
    return np.array(new_paths), np.array(new_labels)

# ============================================================================
# REMOVE NEAR DUPLICATES
# ============================================================================
def remove_near_duplicates(paths, labels, threshold=5):
    hashes = []
    new_paths, new_labels = [], []

    for p, l in zip(paths, labels):
        try:
            h = imagehash.phash(Image.open(p))
        except:
            continue

        duplicate = False
        for existing in hashes:
            if abs(h - existing) < threshold:
                duplicate = True
                break

        if not duplicate:
            hashes.append(h)
            new_paths.append(p)
            new_labels.append(l)

    print("After near-duplicate removal:", len(new_paths))
    return np.array(new_paths), np.array(new_labels)

# ============================================================================
# PREPROCESS
# ============================================================================
def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        return None

    img = cv2.resize(img, (128,128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = preprocess_input(img)
    return img

def load_images(paths, labels):
    X, y = [], []
    for p, l in zip(paths, labels):
        img = preprocess_image(p)
        if img is not None:
            X.append(img)
            y.append(l)
    return np.array(X), np.array(y)

# ============================================================================
# NOISE
# ============================================================================
def add_noise(img):
    noise = np.random.normal(0, 0.02, img.shape)
    img = img + noise
    return np.clip(img, -1, 1)

# ============================================================================
# MODEL
# ============================================================================
def build_model():
    base = MobileNetV2(input_shape=(128,128,3), include_top=False, weights='imagenet')

    base.trainable = True
    for layer in base.layers[:-80]:
        layer.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)

    out = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=base.input, outputs=out)

    model.compile(
        optimizer=Adam(1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model

# ============================================================================
# PLOT
# ============================================================================
def plot_training(h):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(h.history['accuracy'], label='train')
    plt.plot(h.history['val_accuracy'], label='val')
    plt.legend(); plt.title("Accuracy")

    plt.subplot(1,2,2)
    plt.plot(h.history['loss'], label='train')
    plt.plot(h.history['val_loss'], label='val')
    plt.legend(); plt.title("Loss")

    plt.show()

# ============================================================================
# MAIN
# ============================================================================
def main():

    extract_dataset()
    base = find_dataset_path()

    clean_images(os.path.join(base, 'debris'))
    clean_images(os.path.join(base, 'non_debris'))

    # LOAD PATHS
    paths, labels = load_paths()

    # REMOVE DUPLICATES
    paths, labels = remove_exact_duplicates(paths, labels)
    paths, labels = remove_near_duplicates(paths, labels)

    # SPLIT (AFTER CLEANING)
    X_train_p, X_temp_p, y_train, y_temp = train_test_split(
        paths, labels, test_size=0.4, stratify=labels, random_state=SEED
    )

    X_val_p, X_test_p, y_val, y_test = train_test_split(
        X_temp_p, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )

    # LOAD IMAGES
    X_train, y_train = load_images(X_train_p, y_train)
    X_val, y_val = load_images(X_val_p, y_val)
    X_test, y_test = load_images(X_test_p, y_test)

    # ADD NOISE
    X_train = np.array([add_noise(x) for x in X_train])

    print("Train:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))

    datagen = ImageDataGenerator(
        rotation_range=25,
        zoom_range=0.25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    model = build_model()

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=30,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', patience=2)
        ]
    )

    plot_training(history)

    loss, acc, auc_score = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}, AUC: {auc_score:.4f}")

    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int)

    print(classification_report(y_test, y_pred))

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.show()

    model.save("FINAL_MODEL_CLEAN.h5")
    print("✅ Model saved!")

# ============================================================================
if __name__ == "__main__":
    main()