# ============================================================================
# IMPORTS
# ============================================================================
import tensorflow as tf
import os, cv2, zipfile, hashlib, warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.utils import shuffle

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings('ignore')

print("TensorFlow:", tf.__version__)

# ============================================================================
# GPU SETUP
# ============================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print("✅ GPU detected")
else:
    print("⚠️ Running on CPU")

# ============================================================================
# EXTRACT DATASET
# ============================================================================
def extract_dataset():
    if os.path.exists("dataset.zip") and not os.path.exists("dataset"):
        print("📦 Extracting dataset.zip...")
        with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
            zip_ref.extractall("dataset")
        print("✅ Extracted")

# ============================================================================
# FIND DATASET PATH
# ============================================================================
def find_dataset_path(base='dataset'):
    for root, dirs, _ in os.walk(base):
        if 'debris' in dirs and 'non_debris' in dirs:
            print("✅ Dataset found at:", root)
            return root
    raise FileNotFoundError("Dataset folders not found")

# ============================================================================
# HASH FUNCTION (FOR DUPLICATES)
# ============================================================================
def get_image_hash(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64))  # normalize size
    return hashlib.md5(img.tobytes()).hexdigest()

# ============================================================================
# LOAD PATHS + REMOVE DUPLICATES
# ============================================================================
def load_paths_remove_duplicates():
    dataset_path = find_dataset_path()

    paths, labels = [], []
    classes = {'debris':0, 'non_debris':1}

    seen_hashes = set()

    for cls, label in classes.items():
        folder = os.path.join(dataset_path, cls)

        for f in os.listdir(folder):
            p = os.path.join(folder, f)

            h = get_image_hash(p)
            if h is None:
                continue

            # REMOVE DUPLICATES
            if h in seen_hashes:
                continue

            seen_hashes.add(h)
            paths.append(p)
            labels.append(label)

    print(f"✅ Total unique images: {len(paths)}")
    return np.array(paths), np.array(labels)

# ============================================================================
# DATA GENERATOR
# ============================================================================
def data_generator(paths, labels, batch_size=16):

    while True:
        idx = np.random.permutation(len(paths))

        for i in range(0, len(paths), batch_size):
            batch_idx = idx[i:i+batch_size]

            X_batch, y_batch = [], []

            for j in batch_idx:
                img = cv2.imread(paths[j])
                if img is None:
                    continue

                img = cv2.resize(img, (224,224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0

                X_batch.append(img)
                y_batch.append(labels[j])

            yield np.array(X_batch), np.array(y_batch)

# ============================================================================
# LOAD TEST DATA
# ============================================================================
def load_test_data(paths, labels):
    X, y = [], []

    for p, l in zip(paths, labels):
        img = cv2.imread(p)
        if img is None:
            continue

        img = cv2.resize(img, (224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0

        X.append(img)
        y.append(l)

    return np.array(X), np.array(y)

# ============================================================================
# MODEL
# ============================================================================
def build_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(1e-4),
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
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.legend()
    plt.title("Loss")

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
    plt.legend()
    plt.title("ROC Curve")
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

    print("\n📥 Loading & cleaning dataset...")
    paths, labels = load_paths_remove_duplicates()

    paths, labels = shuffle(paths, labels, random_state=42)

    # SPLIT AFTER CLEANING
    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=0.3, stratify=labels, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    train_gen = data_generator(X_train, y_train)
    val_gen = data_generator(X_val, y_val)

    model = build_model()
    model.summary()

    print("\n🚀 Training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=len(X_train)//16,
        validation_steps=len(X_val)//16,
        epochs=15,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(patience=3)
        ]
    )

    plot_training(history)

    print("\n📊 Testing...")
    X_test_img, y_test_img = load_test_data(X_test, y_test)

    loss, acc = model.evaluate(X_test_img, y_test_img)
    print("Test Accuracy:", acc)

    y_prob = model.predict(X_test_img)
    y_pred = (y_prob > 0.5).astype(int)

    print("\n📋 Classification Report:")
    print(classification_report(y_test_img, y_pred))

    plot_confusion(y_test_img, y_pred)
    plot_roc(y_test_img, y_prob)
    plot_pr(y_test_img, y_prob)

    model.save("resnet_model_fixed.h5")
    print("✅ Model saved!")

# ============================================================================
if __name__ == "__main__":
    main()