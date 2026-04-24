# ============================================================================
# IMPORTS
# ============================================================================
import tensorflow as tf
import os, cv2, zipfile, warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from PIL import Image
import imagehash

warnings.filterwarnings('ignore')

print("TensorFlow:", tf.__version__)

# ============================================================================
# GPU
# ============================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print("✅ GPU detected")
else:
    print("⚠️ CPU mode")

# ============================================================================
# EXTRACT
# ============================================================================
def extract_dataset():
    if os.path.exists("dataset.zip") and not os.path.exists("dataset"):
        print("📦 Extracting dataset...")
        with zipfile.ZipFile("dataset.zip", 'r') as z:
            z.extractall("dataset")

# ============================================================================
# FIND DATASET
# ============================================================================
def find_dataset():
    for root, dirs, _ in os.walk("dataset"):
        if "debris" in dirs and "non_debris" in dirs:
            print("✅ Dataset found:", root)
            return root
    raise Exception("Dataset not found")

# ============================================================================
# LOAD PATHS
# ============================================================================
def load_paths():
    base = find_dataset()
    paths, labels = [], []
    classes = {'debris':0, 'non_debris':1}

    for cls, label in classes.items():
        folder = os.path.join(base, cls)
        for f in os.listdir(folder):
            paths.append(os.path.join(folder, f))
            labels.append(label)

    return np.array(paths), np.array(labels)

# ============================================================================
# GROUPING (ANTI-LEAKAGE)
# ============================================================================
def create_groups(paths, labels):
    print("🔍 Grouping similar images...")

    groups, group_labels, hashes = [], [], []

    for p, l in zip(paths, labels):
        try:
            h = imagehash.phash(Image.open(p).convert('RGB'))

            placed = False
            for i, h2 in enumerate(hashes):
                if abs(h - h2) < 5:
                    groups[i].append(p)
                    placed = True
                    break

            if not placed:
                hashes.append(h)
                groups.append([p])
                group_labels.append(l)

        except:
            continue

    print("✅ Groups created:", len(groups))
    return groups, group_labels

# ============================================================================
# FLATTEN
# ============================================================================
def flatten(groups, labels):
    X, y = [], []
    for g, l in zip(groups, labels):
        for p in g:
            X.append(p)
            y.append(l)
    return np.array(X), np.array(y)

# ============================================================================
# DATA PIPELINE
# ============================================================================
def preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224,224])

    # 🔥 IMPORTANT FIX
    img = preprocess_input(img)

    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    return img, label

def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(1000)

    return ds.batch(16).prefetch(tf.data.AUTOTUNE)

# ============================================================================
# MODEL
# ============================================================================
def build_model():
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))

    for layer in base.layers[:-80]:
        layer.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base.input, outputs=out)

    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# ============================================================================
# PLOTS
# ============================================================================
def plot_all(history, y_true, y_pred, y_prob):

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.legend()
    plt.title("Accuracy vs Epoch")

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.legend()
    plt.title("Loss vs Epoch")

    plt.show()

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['Debris','Non-Debris'],
                yticklabels=['Debris','Non-Debris'])
    plt.title("Confusion Matrix")
    plt.show()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    plt.show()

    p, r, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(r, p)
    plt.title("Precision-Recall Curve")
    plt.show()

# ============================================================================
# FIXED GRAD-CAM
# ============================================================================
def grad_cam(model, img_path):

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = preprocess_input(np.expand_dims(img_rgb.astype(np.float32), axis=0))

    # Find last conv layer automatically
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            last_conv = layer.name
            break

    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        loss = preds[:,0]

    grads = tape.gradient(loss, conv_out)

    weights = tf.reduce_mean(grads, axis=(0,1,2))
    conv_out = conv_out[0]

    heatmap = tf.reduce_sum(weights * conv_out, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap, (224,224))

    plt.imshow(img_rgb)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM")
    plt.axis('off')
    plt.show()

# ============================================================================
# MAIN
# ============================================================================
def main():

    extract_dataset()

    print("\n📥 Loading dataset...")
    paths, labels = load_paths()

    groups, g_labels = create_groups(paths, labels)

    # Split by GROUP (anti-leakage)
    g_train, g_temp, y_train, y_temp = train_test_split(
        groups, g_labels, test_size=0.3, stratify=g_labels, random_state=42
    )

    g_val, g_test, y_val, y_test = train_test_split(
        g_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    X_train, y_train = flatten(g_train, y_train)
    X_val, y_val = flatten(g_val, y_val)
    X_test, y_test = flatten(g_test, y_test)

    print("Train:", np.bincount(y_train))
    print("Val:", np.bincount(y_val))
    print("Test:", np.bincount(y_test))

    # Class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    train_ds = make_dataset(X_train, y_train, True)
    val_ds = make_dataset(X_val, y_val)
    test_ds = make_dataset(X_test, y_test)

    model = build_model()
    model.summary()

    print("\n🚀 Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(patience=3)
        ]
    )

    # TEST
    y_true, y_prob = [], []

    for x, y in test_ds:
        p = model.predict(x)
        y_true.extend(y.numpy())
        y_prob.extend(p.flatten())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > 0.5).astype(int)

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred))

    plot_all(history, y_true, y_pred, y_prob)

    # Grad-CAM sample
    grad_cam(model, X_test[0])

    model.save("efficientnet_final_fixed.h5")
    print("✅ Model saved!")

# ============================================================================
if __name__ == "__main__":
    main()