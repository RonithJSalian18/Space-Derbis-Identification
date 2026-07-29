import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models, regularizers
from .base import BaseModelBuilder


class EfficientNetBuilder(BaseModelBuilder):
    """
    Production-ready EfficientNetB0 Transfer Learning Builder.

    Fixes for mode collapse & inverted curves:
    - Entirely freezes base_model during initial construction (base_model.trainable = False).
    - Calls base_model(x, training=False) so BatchNormalization layers stay locked in inference mode.
    - Uses Rescaling(255.0) for normalized [0, 1] inputs before preprocess_input.
    - Classification head: GlobalAveragePooling2D -> Dense(128, relu, l2=1e-4) -> Dropout(0.3) -> Dense(1, sigmoid).
    """

    def build(self) -> tf.keras.Model:
        dropout_rate = self.config.get("dropout_rate", 0.3)
        l2_reg = self.config.get("l2_reg", 1e-4)

        inputs = layers.Input(shape=self.input_shape, name="input_image")

        # 1. Scale inputs from [0, 1] to [0, 255]
        x = layers.Rescaling(255.0, name="scale_to_255")(inputs)

        # 2. Apply EfficientNet preprocess_input inside a Lambda layer
        x = layers.Lambda(lambda t: preprocess_input(t), name="preprocess_input")(x)

        # 3. Instantiate base model
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # 4. Freeze backbone completely during build
        base_model.trainable = False

        # 5. Call base_model with training=False to lock BatchNormalization in inference mode
        x = base_model(x, training=False)

        # 6. Classification Head
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name="dense_head"
        )(x)
        x = layers.Dropout(dropout_rate, name="dropout_head")(x)
        outputs = layers.Dense(1, activation='sigmoid', name="predictions")(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Debris")
        return model


def unfreeze_efficientnet(model: tf.keras.Model, fine_tune_at: int = 30):
    """
    Unfreezes top `fine_tune_at` layers of backbone for Phase 2 fine-tuning,
    while keeping lower layers and all BatchNormalization layers strictly frozen.
    Supports EfficientNet, MobileNetV2, and ResNet50 architectures.
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or any(arch in layer.name.lower() for arch in ["efficientnet", "mobilenet", "resnet"]):
            base_model = layer
            break

    if base_model is None:
        return

    base_model.trainable = True
    num_layers = len(base_model.layers)
    freeze_until = max(0, num_layers - fine_tune_at)

    for i, layer in enumerate(base_model.layers):
        if i < freeze_until or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def build_efficientnet(input_shape=(224, 224, 3), dropout_rate=0.3):
    """Helper function to build uncompiled EfficientNet model."""
    builder = EfficientNetBuilder(input_shape=input_shape, config={"dropout_rate": dropout_rate})
    return builder.build()
