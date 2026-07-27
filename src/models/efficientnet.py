import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models, regularizers


def build_efficientnet(input_shape=(128, 128, 3), dropout_rate=0.5):
    """
    Build Transfer Learning model using EfficientNetB0 architecture with native input preprocessing.
    """
    inputs = layers.Input(shape=input_shape)

    # Apply EfficientNet preprocess_input
    x = preprocess_input(inputs * 255.0)

    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_tensor=x
    )
    base_model.trainable = False  # Freeze pretrained weights

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Debris")
    return model
