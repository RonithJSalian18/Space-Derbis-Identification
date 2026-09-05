import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, regularizers
from .base import BaseModelBuilder


class EfficientNetBuilder(BaseModelBuilder):
    """
    Production-ready Stabilized EfficientNetB0 Transfer Learning Builder.

    - Accepts standard [0, 255] float32 RGB input tensors.
    - Uses native EfficientNet internal scaling (no double rescaling).
    - Entirely freezes base_model during initial construction (base_model.trainable = False).
    - Calls base_model(x, training=False) so BatchNormalization layers stay locked in inference mode.
    - Stabilized head: GAP -> BatchNorm -> Dropout -> Dense(128, relu, he_normal, l2) -> BatchNorm -> Dropout -> Dense(1, sigmoid).
    - Prior bias initialization: b0 ≈ ln(10/1) ≈ 2.3 for 10:1 class imbalance mitigation.
    """

    def build(self) -> tf.keras.Model:
        dropout_rate = self.config.get("dropout_rate", 0.3)
        l2_reg = self.config.get("l2_reg", 1e-4)

        inputs = layers.Input(shape=self.input_shape, name="input_image")

        # Instantiate EfficientNetB0 (includes built-in 1/255 rescaling)
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze backbone during initial construction
        base_model.trainable = False

        # Call with training=False to lock BatchNormalization in inference mode
        x = base_model(inputs, training=False)

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

        model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Debris")
        return model


def unfreeze_efficientnet(model: tf.keras.Model, fine_tune_blocks: int = 2):
    """
    Unfreezes top convolutional blocks (e.g. block7, top_conv, block6) of EfficientNetB0 backbone for Phase 2 fine-tuning,
    while keeping all lower layers and all BatchNormalization layers strictly frozen.
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or any(arch in layer.name.lower() for arch in ["efficientnet", "mobilenet", "resnet"]):
            base_model = layer
            break

    if base_model is None:
        return

    base_model.trainable = True

    # Identify blocks to unfreeze: e.g. 'block7', 'top_conv', 'block6'
    target_prefixes = ["top_conv", "top_bn", "top_activation", "block7"]
    if fine_tune_blocks >= 2:
        target_prefixes.append("block6")
    if fine_tune_blocks >= 3:
        target_prefixes.append("block5")

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        elif any(layer.name.startswith(pfx) for pfx in target_prefixes):
            layer.trainable = True
        else:
            layer.trainable = False


def build_efficientnet(input_shape=(224, 224, 3), dropout_rate=0.3):
    """Helper function to build uncompiled EfficientNet model."""
    builder = EfficientNetBuilder(input_shape=input_shape, config={"dropout_rate": dropout_rate})
    return builder.build()

