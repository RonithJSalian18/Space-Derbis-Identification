"""
Space Debris Detection using CNN (Convolutional Neural Network)
================================================================
Complete Python script for Google Colab that:
1. Extracts dataset from ZIP
2. Loads and preprocesses images (OpenCV pipeline)
3. Builds CNN model (Keras)
4. Trains and evaluates model
5. Saves trained model
6. Provides prediction function

Author: Computer Vision Expert
Platform: Google Colab
Date: 2024
"""

!pip install tensorflow opencv-python scikit-learn seaborn matplotlib

# ============================================================================
# IMPORTS
# ============================================================================

import os
import zipfile
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("✅ All imports successful!")
print(f"TensorFlow version: {tf.__version__}")


# ============================================================================
# STEP 1: EXTRACT DATASET
# ============================================================================

def extract_dataset(zip_path='dataset.zip', extract_to='dataset'):
    """
    Extract dataset.zip and verify folder structure.
    
    Args:
        zip_path (str): Path to ZIP file
        extract_to (str): Directory to extract to
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*70)
    print("STEP 1: EXTRACTING DATASET")
    print("="*70)
    
    try:
        # Check if ZIP file exists
        if not os.path.exists(zip_path):
            print(f"❌ Error: {zip_path} not found!")
            print("Please upload dataset.zip to Colab")
            return False
        
        # Extract ZIP file
        print(f"\n📦 Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"✅ Successfully extracted to {extract_to}/")
        
        # Verify folder structure
        print("\n📂 Verifying folder structure...")
        
        # Check if dataset subfolder exists
        dataset_path = os.path.join(extract_to, 'dataset')
        if not os.path.exists(dataset_path):
            dataset_path = extract_to
        
        folders = ['debris', 'non_debris']
        all_exist = True
        
        for folder in folders:
            folder_path = os.path.join(dataset_path, folder)
            if os.path.exists(folder_path):
                num_files = len([f for f in os.listdir(folder_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                print(f"   ✓ {folder}/ - {num_files} images found")
            else:
                print(f"   ✗ {folder}/ - NOT FOUND!")
                all_exist = False
        
        if all_exist:
            print("\n✅ Dataset structure verified!")
            return True
        else:
            print("\n❌ Dataset structure incomplete!")
            return False
    
    except Exception as e:
        print(f"❌ Error during extraction: {str(e)}")
        return False


# ============================================================================
# STEP 2: IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image_path):
    """
    Apply complete preprocessing pipeline to image:
    1. Resize to (128, 128)
    2. Convert to grayscale
    3. Apply Gaussian Blur (3x3)
    4. Apply CLAHE (clipLimit=2.0, tileGridSize=(8,8))
    5. Apply Canny Edge Detection (50, 150)
    6. Normalize to [0, 1]
    7. Expand dimensions to (128, 128, 1)
    
    Args:
        image_path (str): Path to image file
    
    Returns:
        np.ndarray: Preprocessed image (128, 128, 1) or None if error
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        
        if img is None:
            return None
        
        # Step 1: Resize to (128, 128)
        img_resized = cv2.resize(img, (128, 128))
        
        # Step 2: Convert to grayscale
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Step 3: Apply Gaussian Blur (3x3 kernel)
        img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
        
        # Step 4: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_blurred)
        
        # Step 5: Apply Canny Edge Detection (thresholds: 50, 150)
        img_edges = cv2.Canny(img_clahe, 50, 150)
        
        # Step 6: Normalize to [0, 1]
        img_normalized = img_edges.astype(np.float32) / 255.0
        
        # Step 7: Expand dimensions to (128, 128, 1)
        img_expanded = np.expand_dims(img_normalized, axis=-1)
        
        return img_expanded
    
    except Exception as e:
        return None


# ============================================================================
# STEP 3: LOAD DATASET
# ============================================================================

def load_dataset(dataset_path='dataset'):
    """
    Load all images from dataset folders and preprocess them.
    
    Args:
        dataset_path (str): Path to dataset directory
    
    Returns:
        tuple: (X, y) where X is image array and y is labels
               X shape: (num_samples, 128, 128, 1)
               y shape: (num_samples,) with values 0 (debris) or 1 (non_debris)
    """
    print("\n" + "="*70)
    print("STEP 2: LOADING AND PREPROCESSING IMAGES")
    print("="*70)
    
    X = []  # Images
    y = []  # Labels
    
    # Find dataset directory
    if not os.path.exists(dataset_path):
        dataset_path = 'dataset'
    
    if not os.path.exists(os.path.join(dataset_path, 'dataset')):
        actual_dataset_path = dataset_path
    else:
        actual_dataset_path = os.path.join(dataset_path, 'dataset')
    
    # Define class labels: debris=0, non_debris=1
    class_labels = {'debris': 0, 'non_debris': 1}
    
    # Process each class
    for class_name, label in class_labels.items():
        class_path = os.path.join(actual_dataset_path, class_name)
        
        if not os.path.exists(class_path):
            print(f"⚠️  {class_path} not found, skipping...")
            continue
        
        # Get list of image files
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        print(f"\n📸 Processing {class_name}/ ({len(image_files)} images)")
        
        # Process each image
        successful = 0
        failed = 0
        
        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(class_path, image_file)
            
            # Preprocess image
            preprocessed = preprocess_image(image_path)
            
            if preprocessed is not None:
                X.append(preprocessed)
                y.append(label)
                successful += 1
            else:
                failed += 1
            
            # Print progress
            if idx % max(1, len(image_files) // 5) == 0 or idx == len(image_files):
                print(f"   Progress: {idx}/{len(image_files)} (Failed: {failed})")
        
        print(f"   ✓ Successfully loaded: {successful}/{len(image_files)}")
    
    # Convert to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n✅ Dataset loaded successfully!")
    print(f"   Total samples: {len(X)}")
    print(f"   Image shape: {X.shape}")
    print(f"   Label shape: {y.shape}")
    print(f"   Debris (0): {np.sum(y == 0)}")
    print(f"   Non-debris (1): {np.sum(y == 1)}")
    print(f"   Value range: [{X.min():.2f}, {X.max():.2f}]")
    
    return X, y


# ============================================================================
# STEP 4: SPLIT DATA
# ============================================================================

def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Split strategy:
    - Training: 70%
    - Validation: 15%
    - Test: 15%
    
    Args:
        X (np.ndarray): Feature array
        y (np.ndarray): Label array
        test_size (float): Proportion of test set
        val_size (float): Proportion of validation set
        random_state (int): Random seed
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("\n" + "="*70)
    print("STEP 3: SPLITTING DATA")
    print("="*70)
    
    # First split: separate test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Second split: separate validation from remaining data (15% of remaining)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )
    
    print(f"\n📊 Data split completed:")
    print(f"   Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    print(f"\n   Training - Debris: {np.sum(y_train == 0)}, Non-debris: {np.sum(y_train == 1)}")
    print(f"   Validation - Debris: {np.sum(y_val == 0)}, Non-debris: {np.sum(y_val == 1)}")
    print(f"   Test - Debris: {np.sum(y_test == 0)}, Non-debris: {np.sum(y_test == 1)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# STEP 5: BUILD CNN MODEL
# ============================================================================

def build_model():
    """
    Build CNN model for binary classification (debris vs non-debris).
    
    Architecture:
    - Conv2D(32, (3,3), activation='relu') -> MaxPooling2D(2,2)
    - Conv2D(64, (3,3), activation='relu') -> MaxPooling2D(2,2)
    - Conv2D(128, (3,3), activation='relu') -> MaxPooling2D(2,2)
    - Flatten -> Dense(128, activation='relu') -> Dropout(0.5)
    - Dense(1, activation='sigmoid')
    
    Returns:
        keras.Model: Compiled model
    """
    print("\n" + "="*70)
    print("STEP 4: BUILDING CNN MODEL")
    print("="*70)
    
    model = Sequential([
        # Input layer
        layers.Input(shape=(128, 128, 1)),
        
        # Convolutional Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'),
        MaxPooling2D((2, 2), name='pool1'),
        
        # Convolutional Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        MaxPooling2D((2, 2), name='pool2'),
        
        # Convolutional Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        MaxPooling2D((2, 2), name='pool3'),
        
        # Flatten
        Flatten(name='flatten'),
        
        # Dense layers
        Dense(128, activation='relu', name='dense1'),
        Dropout(0.5, name='dropout'),
        
        # Output layer (Binary classification)
        Dense(1, activation='sigmoid', name='output')
    ])
    
    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    print("\n✅ Model built successfully!")
    print("\n📋 Model Architecture:")
    model.summary()
    
    return model


# ============================================================================
# STEP 6: TRAIN MODEL
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val, epochs=25, batch_size=32):
    """
    Train the CNN model.
    
    Args:
        model (keras.Model): Compiled model
        X_train (np.ndarray): Training images
        y_train (np.ndarray): Training labels
        X_val (np.ndarray): Validation images
        y_val (np.ndarray): Validation labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size
    
    Returns:
        keras.callbacks.History: Training history
    """
    print("\n" + "="*70)
    print("STEP 5: TRAINING MODEL")
    print("="*70)
    
    print(f"\n🚀 Training for {epochs} epochs with batch size {batch_size}...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    print("\n✅ Training completed!")
    
    return history


# ============================================================================
# STEP 7: EVALUATE MODEL
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set.
    
    Args:
        model (keras.Model): Trained model
        X_test (np.ndarray): Test images
        y_test (np.ndarray): Test labels
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*70)
    print("STEP 6: EVALUATING MODEL")
    print("="*70)
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n📊 Test Set Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n📈 Confusion Matrix:")
    print(f"   [[TN={cm[0,0]}  FP={cm[0,1]}]")
    print(f"    [FN={cm[1,0]}  TP={cm[1,1]}]]")
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Debris', 'Non-Debris']))
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n✅ Additional Metrics:")
    print(f"   Sensitivity (Recall): {sensitivity:.4f}")
    print(f"   Specificity: {specificity:.4f}")
    
    metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_test': y_test,
        'y_pred_probs': y_pred_probs
    }
    
    return metrics


# ============================================================================
# STEP 8: PLOT RESULTS
# ============================================================================

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss.
    
    Args:
        history (keras.callbacks.History): Training history
    """
    print("\n" + "="*70)
    print("STEP 7: PLOTTING RESULTS")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy vs Epochs', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss vs Epochs', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\n✅ Training history plot saved as 'training_history.png'")
    plt.show()


def plot_confusion_matrix(cm):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm (np.ndarray): Confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Debris', 'Non-Debris'],
                yticklabels=['Debris', 'Non-Debris'],
                annot_kws={'size': 14})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Confusion matrix plot saved as 'confusion_matrix.png'")
    plt.show()


# ============================================================================
# STEP 9: SAVE MODEL
# ============================================================================

def save_model(model, model_path='debris_cnn_model.h5'):
    """
    Save trained model to disk.
    
    Args:
        model (keras.Model): Trained model
        model_path (str): Path to save model
    """
    print("\n" + "="*70)
    print("STEP 8: SAVING MODEL")
    print("="*70)
    
    try:
        model.save(model_path)
        print(f"\n✅ Model saved successfully as '{model_path}'")
        
        # Print file size
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"   File size: {file_size:.2f} MB")
    
    except Exception as e:
        print(f"❌ Error saving model: {str(e)}")


# ============================================================================
# STEP 10: PREDICTION FUNCTION
# ============================================================================

def predict_image(image_path, model):
    """
    Predict whether an image contains debris or not.
    
    Preprocessing matches training pipeline:
    1. Resize to (128, 128)
    2. Convert to grayscale
    3. Apply Gaussian Blur
    4. Apply CLAHE
    5. Apply Canny Edge Detection
    6. Normalize to [0, 1]
    7. Expand dimensions
    8. Make prediction
    
    Args:
        image_path (str): Path to image file
        model (keras.Model): Trained model
    
    Returns:
        dict: Prediction result with confidence
    """
    try:
        # Preprocess image
        preprocessed = preprocess_image(image_path)
        
        if preprocessed is None:
            return {'error': 'Could not read or process image'}
        
        # Add batch dimension
        input_data = np.expand_dims(preprocessed, axis=0)
        
        # Make prediction
        prediction_prob = model.predict(input_data, verbose=0)[0][0]
        
        # Convert to class label
        prediction_class = 'Non-Debris' if prediction_prob > 0.5 else 'Debris'
        confidence = prediction_prob if prediction_prob > 0.5 else (1 - prediction_prob)
        
        return {
            'image_path': image_path,
            'prediction': prediction_class,
            'confidence': float(confidence),
            'probability_debris': float(1 - prediction_prob),
            'probability_non_debris': float(prediction_prob),
            'success': True
        }
    
    except Exception as e:
        return {'error': f'Prediction error: {str(e)}', 'success': False}


# ============================================================================
# STEP 11: VISUALIZE SAMPLE PREDICTIONS
# ============================================================================

def visualize_predictions(X_test, y_test, y_pred, num_samples=9):
    """
    Visualize sample predictions with original (edge-detected) images.
    
    Args:
        X_test (np.ndarray): Test images
        y_test (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        num_samples (int): Number of samples to visualize
    """
    print("\n📸 Visualizing sample predictions...")
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.ravel()
    
    # Select random samples
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    
    for idx, sample_idx in enumerate(indices):
        ax = axes[idx]
        
        # Get image
        img = X_test[sample_idx, :, :, 0]  # Remove channel dimension
        
        # Get labels
        true_label = 'Debris' if y_test[sample_idx] == 0 else 'Non-Debris'
        pred_label = 'Debris' if y_pred[sample_idx] == 0 else 'Non-Debris'
        
        # Color based on correctness
        correct = true_label == pred_label
        color = 'green' if correct else 'red'
        
        # Display image
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(indices), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
    print("✅ Sample predictions saved as 'sample_predictions.png'")
    plt.show()


# ============================================================================
# STEP 12: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - runs the complete pipeline.
    """
    print("\n" + "="*70)
    print("   CNN-BASED SPACE DEBRIS DETECTION SYSTEM")
    print("   Platform: Google Colab")
    print("="*70)
    
    # ===== STEP 1: Extract Dataset =====
    dataset_extracted = extract_dataset()
    if not dataset_extracted:
        print("\n❌ Failed to extract dataset. Exiting.")
        return
    
    # ===== STEP 2: Load Dataset =====
    X, y = load_dataset()
    if X is None or len(X) == 0:
        print("\n❌ Failed to load dataset. Exiting.")
        return
    
    # ===== STEP 3: Split Data =====
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # ===== STEP 4: Build Model =====
    model = build_model()
    
    # ===== STEP 5: Train Model =====
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=25, batch_size=32)
    
    # ===== STEP 6: Evaluate Model =====
    metrics = evaluate_model(model, X_test, y_test)
    
    # ===== STEP 7: Plot Results =====
    plot_training_history(history)
    plot_confusion_matrix(metrics['confusion_matrix'])
    
    # ===== STEP 8: Save Model =====
    save_model(model, 'debris_cnn_model.h5')
    
    # ===== STEP 9: Visualize Predictions =====
    visualize_predictions(X_test, y_test, metrics['y_pred'], num_samples=9)
    
    # ===== STEP 10: Test Prediction Function =====
    print("\n" + "="*70)
    print("STEP 9: TESTING PREDICTION FUNCTION")
    print("="*70)
    
    print("\n🧪 Testing prediction on random test samples...")
    
    for i in range(min(3, len(X_test))):
        # Create temporary image file from test sample
        test_img = X_test[i, :, :, 0]
        
        # Denormalize and convert to uint8
        test_img_uint8 = (test_img * 255).astype(np.uint8)
        
        # Save temporarily
        cv2.imwrite('temp_test_image.png', test_img_uint8)
        
        # Predict
        result = predict_image('temp_test_image.png', model)
        
        if result['success']:
            true_label = 'Debris' if y_test[i] == 0 else 'Non-Debris'
            pred_label = result['prediction']
            confidence = result['confidence']
            
            match = "✅" if true_label == pred_label else "❌"
            print(f"\n{match} Test {i+1}:")
            print(f"   True label: {true_label}")
            print(f"   Prediction: {pred_label}")
            print(f"   Confidence: {confidence:.2%}")
            print(f"   Debris prob: {result['probability_debris']:.4f}")
            print(f"   Non-Debris prob: {result['probability_non_debris']:.4f}")
    
    # Clean up
    if os.path.exists('temp_test_image.png'):
        os.remove('temp_test_image.png')
    
    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Test Accuracy: {metrics['test_accuracy']:.2%}")
    print(f"   Test Loss: {metrics['test_loss']:.4f}")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   • debris_cnn_model.h5 (Trained model)")
    print(f"   • training_history.png (Training curves)")
    print(f"   • confusion_matrix.png (Performance matrix)")
    print(f"   • sample_predictions.png (Prediction examples)")
    
    print(f"\n💾 TO LOAD MODEL LATER:")
    print(f"   from tensorflow.keras.models import load_model")
    print(f"   model = load_model('debris_cnn_model.h5')")
    print(f"   result = predict_image('image.jpg', model)")
    
    print(f"\n🎉 Ready for deployment!")
    
    return model, metrics


# ============================================================================
# RUN MAIN FUNCTION
# ============================================================================

if __name__ == "__main__":
    model, metrics = main()