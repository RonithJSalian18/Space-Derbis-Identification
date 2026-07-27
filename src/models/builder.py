import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from configs import LEARNING_RATE, IMAGE_SIZE, LABEL_SMOOTHING
from .cnn import build_custom_cnn
from .mobilenet import build_mobilenet
from .resnet import build_resnet
from .efficientnet import build_efficientnet


def get_model(architecture_name="cnn", learning_rate=LEARNING_RATE):
    """
    Factory function to instantiate and compile model by name.
    Supported architectures: 'cnn', 'mobilenet', 'resnet', 'efficientnet'
    """
    arch = architecture_name.lower()
    h, w = IMAGE_SIZE

    if arch == "cnn":
        model = build_custom_cnn(input_shape=(h, w, 1))
        color_mode = "grayscale"
    elif arch in ["mobilenet", "mobile"]:
        model = build_mobilenet(input_shape=(h, w, 3))
        color_mode = "rgb"
    elif arch in ["resnet", "res"]:
        model = build_resnet(input_shape=(h, w, 3))
        color_mode = "rgb"
    elif arch in ["efficientnet", "effinet"]:
        model = build_efficientnet(input_shape=(h, w, 3))
        color_mode = "rgb"
    else:
        raise ValueError(f"Unknown architecture '{architecture_name}'. Supported: cnn, mobilenet, resnet, efficientnet")

    # Regularized Binary Crossentropy with Label Smoothing to prevent overconfidence/overfitting
    loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    return model, color_mode
