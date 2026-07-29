import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from .base import BaseModelBuilder


class CustomCNNBuilder(BaseModelBuilder):
    """Custom Convolutional Neural Network Builder optimized for Space Debris Classification."""

    def build(self) -> tf.keras.Model:
        l2_reg = self.config.get("l2_reg", 0.001)
        dropout_rate = self.config.get("dropout_rate", 0.5)

        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                          kernel_regularizer=regularizers.l2(l2_reg), input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                          kernel_regularizer=regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                          kernel_regularizer=regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                          kernel_regularizer=regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Global Average Pooling & Dense Head
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(0.0))
        ], name="Custom_CNN_Debris")

        return model


def build_custom_cnn(input_shape=(224, 224, 1), l2_reg=0.001, dropout_rate=0.5):
    """Legacy helper function for custom CNN construction."""
    builder = CustomCNNBuilder(input_shape=input_shape, config={"l2_reg": l2_reg, "dropout_rate": dropout_rate})
    return builder.build()
