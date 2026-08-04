"""
Zero-Trust Grad-CAM Visual Audit for Space Debris Identification System.

Performs Gradient-weighted Class Activation Mapping (Grad-CAM) visual audits on trained
Keras CNN models (EfficientNetB0, ResNet50, MobileNetV2, Custom CNN) to verify that 
predictions focus on physical spacecraft geometry rather than background space or artifacts.

Features:
- Dynamic discovery of the last convolutional layer across any architecture.
- Gradient computation w.r.t. predicted class activations.
- Heatmap generation & seamless overlay on original imagery.
- Standalone CLI execution saving visual audit plots with colorbars to plots/ directory.

Usage:
    python -m src.evaluation.gradcam --model saved_models/cnn_spark_debris.h5 --image sample_debris/img022768.jpg
"""

import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from configs import CLASS_NAMES
from src.data.preprocessing import preprocess_image


def find_last_conv_layer(model: tf.keras.Model) -> tuple:
    """
    Dynamically identifies the last Conv2D layer in any Keras model or nested backbone model.

    Returns:
        tuple: (backbone_layer_name, last_conv_layer_name)
    """
    # 1. Search for nested backbone layer (e.g., EfficientNet, ResNet, MobileNet)
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model) or any(arch in layer.name.lower() for arch in ["efficientnet", "mobilenet", "resnet"]):
            backbone_name = layer.name
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D) or "top_conv" in sub_layer.name or "conv" in sub_layer.name:
                    return backbone_name, sub_layer.name

    # 2. Search direct layers if no nested backbone wrapper exists
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return None, layer.name

    raise ValueError("❌ No Conv2D or convolutional backbone layer found in the provided model.")


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    backbone_layer_name: str = None,
    last_conv_layer_name: str = None,
    pred_index: int = None
) -> np.ndarray:
    """
    Computes Grad-CAM heatmap for an input image array.

    Args:
        img_array (np.ndarray): Preprocessed image tensor (H, W, C) or (1, H, W, C).
        model (tf.keras.Model): Trained Keras model.
        backbone_layer_name (str): Name of nested backbone layer if applicable.
        last_conv_layer_name (str): Name of target Conv2D layer.
        pred_index (int): Class index to compute gradients for (defaults to top prediction).

    Returns:
        np.ndarray: Normalized 2D heatmap matrix [0, 1].
    """
    if len(img_array.shape) == 3:
        img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0), dtype=tf.float32)
    else:
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    if backbone_layer_name is None and last_conv_layer_name is None:
        backbone_layer_name, last_conv_layer_name = find_last_conv_layer(model)

    if backbone_layer_name:
        backbone = model.get_layer(backbone_layer_name)
        if last_conv_layer_name is None:
            _, last_conv_layer_name = find_last_conv_layer(backbone)

        target_conv_layer = backbone.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=backbone.inputs,
            outputs=[target_conv_layer.output, backbone.output]
        )

        # Forward pass through initial preprocessing layers before backbone
        x = img_tensor
        for l in model.layers:
            if l.name == backbone_layer_name:
                break
            x = l(x)

        with tf.GradientTape() as tape:
            conv_outputs, backbone_features = grad_model(x)
            tape.watch(conv_outputs)

            # Pass features through classification head layers
            h = backbone_features
            start_idx = model.layers.index(model.get_layer(backbone_layer_name)) + 1
            for l in model.layers[start_idx:]:
                h = l(h)
            predictions = h

            if pred_index is None:
                pred_index = 0 if predictions[0][0] <= 0.5 else 1

            loss = predictions[:, 0] if pred_index == 0 else (1.0 - predictions[:, 0])

        grads = tape.gradient(loss, conv_outputs)
    else:
        target_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            tape.watch(conv_outputs)

            if pred_index is None:
                pred_index = 0 if predictions[0][0] <= 0.5 else 1

            loss = predictions[:, 0] if pred_index == 0 else (1.0 - predictions[:, 0])

        grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        return np.zeros((img_tensor.shape[1], img_tensor.shape[2]), dtype=np.float32)

    # Guided pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def run_zero_trust_audit(
    model_path: str,
    image_path: str,
    output_dir: str = "plots/gradcam_audit",
    color_mode: str = "rgb",
    model_type: str = "cnn"
) -> str:
    """
    Executes a complete zero-trust Grad-CAM audit on a single image file, saving plot with colorbar.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n==================================================")
    print(f"[+] ZERO-TRUST GRAD-CAM AUDIT ENGINE")
    print(f"Model Path: {model_path}")
    print(f"Input Image: {image_path}")
    print(f"==================================================")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[-] Model file not found at: {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[-] Input image not found at: {image_path}")

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Full model architecture and weights loaded successfully.")
    except Exception:
        from src.models import ModelFactory
        print("[+] Model file contains weights only. Instantiating architecture via ModelFactory...")
        model, _ = ModelFactory.create_model(architecture_name=model_type)
        model.load_weights(model_path)
        print("✅ Model weights loaded into architecture successfully.")

    backbone_name, last_conv_name = find_last_conv_layer(model)
    print(f"[+] Target Conv Layer Identified: Backbone='{backbone_name}', Layer='{last_conv_name}'")

    # Load & preprocess image
    img_tensor = preprocess_image(image_path, color_mode=color_mode, target_size=(224, 224), model_type=model_type)
    if img_tensor is None:
        raise ValueError(f"Could not load/preprocess image from {image_path}")

    prob_non_debris = float(model.predict(np.expand_dims(img_tensor, axis=0), verbose=0)[0][0])
    prob_debris = 1.0 - prob_non_debris

    if prob_non_debris > 0.5:
        pred_label = "Non-Debris"
        confidence = prob_non_debris * 100.0
    else:
        pred_label = "Debris"
        confidence = prob_debris * 100.0

    heatmap = make_gradcam_heatmap(img_tensor, model, backbone_name, last_conv_name)

    # Prepare display RGB image
    if img_tensor.shape[-1] == 1:
        img_rgb = np.repeat(img_tensor, 3, axis=-1)
    else:
        img_rgb = img_tensor.copy()

    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

    overlay = cv2.addWeighted(np.float32(img_rgb), 0.6, np.float32(heatmap_color), 0.4, 0)

    # Create figure with colorbar
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Space Image (224x224)", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    im_hm = axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title("Grad-CAM Activation Heatmap", fontsize=11, fontweight='bold')
    axes[1].axis('off')
    cbar = fig.colorbar(im_hm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Activation Intensity", fontsize=10)

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay Audit (Pred: {pred_label} - {confidence:.1f}%)", fontsize=11, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle(f"Zero-Trust Grad-CAM Audit | Layer: {last_conv_name} | Class: {pred_label}", fontsize=13, fontweight='bold')
    plt.tight_layout()

    filename = os.path.basename(image_path)
    out_path = os.path.abspath(os.path.join(output_dir, f"gradcam_audit_{filename}.png"))
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"\n==================================================")
    print(f"✅ Grad-CAM Audit Plot Saved to: {out_path}")
    print(f"Prediction: {pred_label} ({confidence:.2f}% confidence)")
    print(f"==================================================")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Zero-Trust Grad-CAM Visual Audit for Space Debris Identification")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model .h5 file")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--model-type", type=str, default="cnn", choices=["cnn", "mobilenet", "resnet", "efficientnet"], help="Model architecture type")
    parser.add_argument("--output-dir", type=str, default="plots/gradcam_audit", help="Output directory for audit plots")
    parser.add_argument("--color-mode", type=str, default="rgb", choices=["rgb", "grayscale"], help="Input image color mode")

    args = parser.parse_args()
    run_zero_trust_audit(model_path=args.model, image_path=args.image, output_dir=args.output_dir, color_mode=args.color_mode, model_type=args.model_type)


if __name__ == "__main__":
    main()
