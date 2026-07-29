import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models, regularizers
from .base import BaseModelBuilder


class MobileNetBuilder(BaseModelBuilder):
    """
    Production-ready MobileNetV2 Transfer Learning Builder.

    - Entirely freezes base_model during initial construction (base_model.trainable = False).
    - Calls base_model(x, training=False) so BatchNormalization layers stay locked in inference mode.
    - Rescaling(255.0) for normalized [0, 1] inputs before preprocess_input.
    - Classification head: GAP -> Dense(128, relu, l2=1e-4) -> Dropout(0.3) -> Dense(1, sigmoid).
    """

    def build(self) -> tf.keras.Model:
        dropout_rate = self.config.get("dropout_rate", 0.3)
        l2_reg = self.config.get("l2_reg", 1e-4)

        inputs = layers.Input(shape=self.input_shape, name="input_image")
        x = layers.Rescaling(255.0, name="scale_to_255")(inputs)
        x = layers.Lambda(lambda t: preprocess_input(t), name="preprocess_input")(x)

        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        base_model.trainable = False

        x = base_model(x, training=False)

        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name="dense_head"
        )(x)
        x = layers.Dropout(dropout_rate, name="dropout_head")(x)
        outputs = layers.Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(0.0), name="predictions")(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Debris")
        return model


def build_mobilenet(input_shape=(224, 224, 3), dropout_rate=0.3):
    """Helper function for MobileNet construction."""
    builder = MobileNetBuilder(input_shape=input_shape, config={"dropout_rate": dropout_rate})
    return builder.build()
