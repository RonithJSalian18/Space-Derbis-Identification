import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models, regularizers
from .base import BaseModelBuilder


class MobileNetBuilder(BaseModelBuilder):
    """
    Production-ready Stabilized MobileNetV2 Transfer Learning Builder.

    - Accepts standard [0, 255] float32 RGB input tensors.
    - Applies MobileNetV2 [-1, 1] normalization cleanly without double scaling.
    - Freezes base_model during initial construction (base_model.trainable = False).
    - Calls base_model(x, training=False) so BatchNormalization layers stay locked in inference mode.
    - Stabilized head: GAP -> BatchNorm -> Dropout -> Dense(128, relu, he_normal, l2) -> BatchNorm -> Dropout -> Dense(1, sigmoid).
    - Prior bias initialization: b0 ≈ ln(10/1) ≈ 2.3 for 10:1 class imbalance mitigation.
    """

    def build(self) -> tf.keras.Model:
        dropout_rate = self.config.get("dropout_rate", 0.3)
        l2_reg = self.config.get("l2_reg", 1e-4)

        inputs = layers.Input(shape=self.input_shape, name="input_image")
        
        # Clean preprocess_input: scales [0, 255] to [-1, 1]
        x = layers.Lambda(lambda t: preprocess_input(tf.cast(t, tf.float32)), name="preprocess_input")(inputs)

        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        base_model.trainable = False

        # Call with training=False to lock BatchNormalization in inference mode
        x = base_model(x, training=False)

        # Stabilized Classification Head
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.BatchNormalization(name="head_bn1")(x)
        x = layers.Dropout(dropout_rate, name="head_dropout1")(x)
        x = layers.Dense(
            128,
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(l2_reg),
            name="dense_head"
        )(x)
        x = layers.BatchNormalization(name="head_bn2")(x)
        x = layers.Dropout(dropout_rate * 0.67, name="head_dropout2")(x)
        outputs = layers.Dense(
            1,
            activation='sigmoid',
            bias_initializer=tf.keras.initializers.Constant(2.3),
            name="predictions"
        )(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Debris")
        return model


def unfreeze_mobilenet(model: tf.keras.Model, fine_tune_blocks: int = 2):
    """
    Unfreezes top inverted residual blocks (e.g. block_16, Conv_1) of MobileNetV2 for Phase 2 fine-tuning,
    while keeping lower layers and all BatchNormalization layers strictly frozen.
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        return

    base_model.trainable = True

    target_prefixes = ["Conv_1", "block_16", "out_relu"]
    if fine_tune_blocks >= 2:
        target_prefixes.extend(["block_15", "block_14"])

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        elif any(layer.name.startswith(pfx) for pfx in target_prefixes):
            layer.trainable = True
        else:
            layer.trainable = False


def build_mobilenet(input_shape=(224, 224, 3), dropout_rate=0.3):
    """Helper function for MobileNet construction."""
    builder = MobileNetBuilder(input_shape=input_shape, config={"dropout_rate": dropout_rate})
    return builder.build()

