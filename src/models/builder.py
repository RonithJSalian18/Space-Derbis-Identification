import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from configs import LEARNING_RATE, IMAGE_SIZE, LABEL_SMOOTHING
from .cnn import build_custom_cnn
from .mobilenet import build_mobilenet
from .resnet import build_resnet
from .efficientnet import build_efficientnet


def get_model(architecture_name="cnn", learning_rate=None):
    """
    Factory function to instantiate and compile model by name with Gradient Clipping (clipnorm=1.0)
    and architecture-specific learning rates to prevent exploding loss spikes > 200.
    """
    arch = architecture_name.lower()
    h, w = IMAGE_SIZE

    if arch == "cnn":
        model = build_custom_cnn(input_shape=(h, w, 1))
        color_mode = "grayscale"
        default_lr = 0.0001 if learning_rate is None else learning_rate
    elif arch in ["mobilenet", "mobile"]:
        model = build_mobilenet(input_shape=(h, w, 3))
        color_mode = "rgb"
        # Pretrained transfer learning requires smaller fine-tuning learning rate (1e-5)
        default_lr = 0.00001 if learning_rate is None else learning_rate
    elif arch in ["resnet", "res"]:
        model = build_resnet(input_shape=(h, w, 3))
        color_mode = "rgb"
        default_lr = 0.00001 if learning_rate is None else learning_rate
    elif arch in ["efficientnet", "effinet"]:
        model = build_efficientnet(input_shape=(h, w, 3))
        color_mode = "rgb"
        default_lr = 0.00001 if learning_rate is None else learning_rate
    else:
        raise ValueError(f"Unknown architecture '{architecture_name}'. Supported: cnn, mobilenet, resnet, efficientnet")

    # Regularized Binary Crossentropy with Label Smoothing to prevent overconfidence/overfitting
    loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)

    # Optimizer with Gradient Clipping (clipnorm=1.0) to eliminate wild loss spikes > 200
    optimizer = Adam(learning_rate=default_lr, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    return model, color_mode
