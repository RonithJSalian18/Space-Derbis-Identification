# ============================================================================
# IMPORTS
# ============================================================================
import tensorflow as tf
import os, zipfile, cv2, warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import imagehash  # pip install ImageHash

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.utils import shuffle

from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings('ignore')

# ============================================================================
# GPU CHECK
# ============================================================================
print("\n🔍 Checking GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ GPU ON")
else:
    print("⚠️ CPU MODE")

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
# LOAD DATASET
# ============================================================================
def load_dataset(base_path='dataset'):
    X, y, paths = [], [], []

    data_path = None
    for root, dirs, _ in os.walk(base_path):
        if 'debris' in dirs and 'non_debris' in dirs:
            data_path = root
            break

    if data_path is None:
        print("❌ Dataset not found!")
        return [], [], []

    print("✅ Using:", data_path)

    classes = {'debris':0, 'non_debris':1}

    for cls, label in classes.items():
        folder = os.path.join(data_path, cls)

        for f in os.listdir(folder):
            p = os.path.join(folder, f)
            img = preprocess_image(p)
            if img is not None:
                X.append(img)
                y.append(label)
                paths.append(p)

    return np.array(X), np.array(y), np.array(paths)

# ============================================================================
# REMOVE SIMILAR IMAGES
# ============================================================================
def remove_similar_images(paths, threshold=2):
    print("\n🧹 Removing similar images...")

    hashes = []
    keep_idx = []

    for i, p in enumerate(paths):
        try:
            img = Image.open(p)
            h = imagehash.phash(img)

            duplicate = False
            for existing in hashes:
                if abs(h - existing) <= threshold:
                    duplicate = True
                    break

            if not duplicate:
                hashes.append(h)
                keep_idx.append(i)
        except:
            continue

    print(f"Before: {len(paths)} | After: {len(keep_idx)}")
    return keep_idx

# ============================================================================
# GROUPS (FOR LEAKAGE PREVENTION)
# ============================================================================
def create_groups(paths):
    groups = []
    for p in paths:
        name = os.path.basename(p)
        group_id = name.split('_')[0]
        groups.append(group_id)
    return np.array(groups)

# ============================================================================
# MODEL
# ============================================================================
def build_model():
    model = Sequential([
        layers.Input(shape=(128,128,1)),

        Conv2D(32,3,activation='relu'),
        MaxPooling2D(),

        Conv2D(64,3,activation='relu'),
        MaxPooling2D(),

        Flatten(),

        Dense(64, activation='relu',
              kernel_regularizer=regularizers.l2(0.001)),

        Dropout(0.6),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ============================================================================
# TRAINING PLOTS
# ============================================================================
def plot_training(history):
    plt.style.use('seaborn-v0_8')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc)+1)

    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_graph.png", dpi=300)
    plt.show()

# ============================================================================
# OVERLAP CHECK
# ============================================================================
def check_overlap(train_paths, test_paths):
    overlap = set(train_paths).intersection(set(test_paths))
    print("\n🔍 Overlap between train & test:", len(overlap))

# ============================================================================
# MAIN
# ============================================================================
def main():

    extract_dataset()

    print("\n📥 Loading...")
    X, y, paths = load_dataset()

    if len(X) == 0:
        return

    # Remove similar images
    keep_idx = remove_similar_images(paths, threshold=2)
    X, y, paths = X[keep_idx], y[keep_idx], paths[keep_idx]

    X, y, paths = shuffle(X, y, paths, random_state=42)

    # Group split
    groups = create_groups(paths)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(gss.split(X, y, groups))

    X_train, X_temp = X[train_idx], X[temp_idx]
    y_train, y_temp = y[train_idx], y[temp_idx]
    paths_train, paths_temp = paths[train_idx], paths[temp_idx]

    groups_temp = groups[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups_temp))

    X_val, X_test = X_temp[val_idx], X_temp[test_idx]
    y_val, y_test = y_temp[val_idx], y_temp[test_idx]
    paths_val, paths_test = paths_temp[val_idx], paths_temp[test_idx]

    print("\n📊 Dataset Split:")
    print("Train:", len(X_train))
    print("Val:", len(X_val))
    print("Test:", len(X_test))

    check_overlap(paths_train, paths_test)

    # Augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    model = build_model()

    # Train
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=30,
        callbacks=[
            EarlyStopping(patience=7, restore_best_weights=True),
            ReduceLROnPlateau(patience=4)
        ]
    )

    # 🔥 PLOT TRAINING GRAPHS
    plot_training(history)

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print("\n✅ Test Accuracy:", acc)

    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int)

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['Debris','Non-Debris'],
                yticklabels=['Debris','Non-Debris'])
    plt.title("Confusion Matrix")
    plt.show()

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.title("ROC Curve")
    plt.show()

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.show()

    model.save("final_debris_model_fixed.h5")
    print("✅ Model saved")

# ============================================================================
if __name__ == "__main__":
    main()