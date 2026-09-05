import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models, regularizers
from .base import BaseModelBuilder


class ResNetBuilder(BaseModelBuilder):
    """
    Production-ready Stabilized ResNet50 Transfer Learning Builder.

    - Accepts standard [0, 255] float32 RGB input tensors.
    - Applies ResNet-50 mean subtraction & BGR conversion cleanly without double scaling.
    - Freezes base_model during construction (base_model.trainable = False).
    - Calls base_model(x, training=False) so BatchNormalization layers stay locked in inference mode.
    - Stabilized head: GAP -> BatchNorm -> Dropout -> Dense(128, relu, he_normal, l2) -> BatchNorm -> Dropout -> Dense(1, sigmoid).
    - Prior bias initialization: b0 ≈ ln(10/1) ≈ 2.3 for 10:1 class imbalance mitigation.
    """

    def build(self) -> tf.keras.Model:
        dropout_rate = self.config.get("dropout_rate", 0.3)
        l2_reg = self.config.get("l2_reg", 1e-4)

        inputs = layers.Input(shape=self.input_shape, name="input_image")
        
        # Clean preprocess_input: converts RGB -> BGR and applies ImageNet zero-centering
        x = layers.Lambda(lambda t: preprocess_input(tf.cast(t, tf.float32)), name="preprocess_input")(inputs)

        base_model = ResNet50(
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

        model = models.Model(inputs=inputs, outputs=outputs, name="ResNet50_Debris")
        return model


def unfreeze_resnet(model: tf.keras.Model, fine_tune_stage: str = "conv5"):
    """
    Unfreezes complete residual stage (e.g. stage 5: 'conv5') of ResNet50 backbone for Phase 2 fine-tuning,
    preserving all residual skip connections while keeping all BatchNormalization layers strictly frozen.
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "resnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        return

    base_model.trainable = True

    # Selectively unfreeze only target stage conv layers, locking BatchNormalization
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        elif fine_tune_stage in layer.name.lower() or "conv5" in layer.name.lower():
            layer.trainable = True
        else:
            layer.trainable = False


def build_resnet(input_shape=(224, 224, 3), dropout_rate=0.3):
    """Helper function for ResNet50 construction."""
    builder = ResNetBuilder(input_shape=input_shape, config={"dropout_rate": dropout_rate})
    return builder.build()

