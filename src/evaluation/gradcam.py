import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from configs import CLASS_NAMES


def get_last_conv_layer_name(model):
    """Find the last Conv2D or functional backbone layer in the Keras model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model) or "efficientnet" in layer.name.lower() or "mobilenet" in layer.name.lower() or "resnet" in layer.name.lower():
            # Search inside the nested backbone model
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D) or "top_conv" in sub_layer.name or "conv" in sub_layer.name:
                    return layer.name, sub_layer.name
        elif isinstance(layer, tf.keras.layers.Conv2D):
            return None, layer.name

    return None, None


def make_gradcam_heatmap(img_array, model, backbone_layer_name=None, last_conv_layer_name=None):
    """
    Compute Grad-CAM heatmap for a given input image array.
    """
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    if backbone_layer_name:
        backbone = model.get_layer(backbone_layer_name)
        if last_conv_layer_name is None:
            for l in reversed(backbone.layers):
                if isinstance(l, tf.keras.layers.Conv2D) or "top_conv" in l.name or "conv" in l.name:
                    last_conv_layer_name = l.name
                    break

        grad_model = tf.keras.models.Model(
            inputs=backbone.inputs,
            outputs=[backbone.get_layer(last_conv_layer_name).output, backbone.output]
        )

        # Forward pass through preprocessing & scaling layers before backbone
        x = img_tensor
        for l in model.layers:
            if l.name == backbone_layer_name:
                break
            x = l(x)

        with tf.GradientTape() as tape:
            conv_outputs, backbone_features = grad_model(x)
            tape.watch(conv_outputs)

            # Pass through GAP and classification head
            h = backbone_features
            for l in model.layers[model.layers.index(model.get_layer(backbone_layer_name)) + 1:]:
                h = l(h)
            predictions = h

        grads = tape.gradient(predictions, conv_outputs)
    else:
        if last_conv_layer_name is None:
            for l in reversed(model.layers):
                if isinstance(l, tf.keras.layers.Conv2D):
                    last_conv_layer_name = l.name
                    break

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            tape.watch(conv_outputs)

        grads = tape.gradient(predictions, conv_outputs)

    if grads is None:
        return np.zeros((img_array.shape[1], img_array.shape[2]), dtype=np.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def save_gradcam_visualizations(model, X_test, y_test, class_names=CLASS_NAMES, save_dir="plots/gradcam", num_samples=6):
    """
    Generate and save Grad-CAM visual overlays for test samples.
    """
    os.makedirs(save_dir, exist_ok=True)
    backbone_name, last_conv_name = get_last_conv_layer_name(model)

    print(f"[+] Generating Grad-CAM heatmaps (Target Backbone: '{backbone_name}', Layer: '{last_conv_name}')...")

    # Select samples from both classes
    debris_idxs = np.where(y_test == 0)[0][:num_samples // 2]
    non_debris_idxs = np.where(y_test == 1)[0][:num_samples // 2]
    sample_idxs = np.concatenate([debris_idxs, non_debris_idxs])

    saved_paths = []
    for i, idx in enumerate(sample_idxs):
        img_arr = X_test[idx]
        true_label = class_names[int(y_test[idx])]

        pred_prob = model.predict(np.expand_dims(img_arr, axis=0), verbose=0)[0][0]
        pred_label = class_names[int(pred_prob > 0.5)]

        try:
            heatmap = make_gradcam_heatmap(img_arr, model, backbone_name, last_conv_name)
        except Exception as e:
            print(f"[-] Grad-CAM computation warning for sample {i}: {e}")
            heatmap = np.zeros((img_arr.shape[0], img_arr.shape[1]), dtype=np.float32)

        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (img_arr.shape[1], img_arr.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

        # Prepare RGB image
        if img_arr.shape[-1] == 1:
            img_rgb = np.repeat(img_arr, 3, axis=-1)
        else:
            img_rgb = img_arr.copy()

        overlay = cv2.addWeighted(np.float32(img_rgb), 0.6, np.float32(heatmap_color), 0.4, 0)

        # Plot 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Original ({true_label})")
        axes[0].axis('off')

        axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay (Pred: {pred_label} - {pred_prob:.2f})")
        axes[2].axis('off')

        plt.suptitle(f"Sample #{i+1} | True: {true_label} | Pred: {pred_label} ({pred_prob:.2f})", fontsize=12, fontweight='bold')
        plt.tight_layout()

        out_path = os.path.abspath(os.path.join(save_dir, f"gradcam_sample_{i+1}_{true_label.lower()}.png"))
        plt.savefig(out_path, dpi=300)
        plt.close()
        saved_paths.append(out_path)

    print(f"[+] Saved {len(saved_paths)} Grad-CAM visualizations to: {os.path.abspath(save_dir)}")
    return saved_paths
