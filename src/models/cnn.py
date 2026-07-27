import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_custom_cnn(input_shape=(224, 224, 1), l2_reg=0.001, dropout_rate=0.5):
    """
    Build custom CNN architecture optimized for Space Debris Binary Classification
    using Global Average Pooling to prevent spatial parameter exploding and overfitting.
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg), input_shape=input_shape),
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

        # Block 4 (Extra receptive field for fine detail in orbital space shots)
        layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Global Average Pooling (drastically reduces overfitting vs raw Flatten)
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ], name="Custom_CNN_Debris")

    return model
