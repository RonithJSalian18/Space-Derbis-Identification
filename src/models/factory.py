import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from typing import Dict, Type, Tuple
from configs import IMAGE_SIZE, LABEL_SMOOTHING
from .base import BaseModelBuilder
from .cnn import CustomCNNBuilder
from .mobilenet import MobileNetBuilder
from .resnet import ResNetBuilder
from .efficientnet import EfficientNetBuilder


class ModelFactory:
    """Factory Pattern registry to build and compile models by name."""

    _registry: Dict[str, Type[BaseModelBuilder]] = {
        "cnn": CustomCNNBuilder,
        "custom_cnn": CustomCNNBuilder,
        "mobilenet": MobileNetBuilder,
        "mobile": MobileNetBuilder,
        "resnet": ResNetBuilder,
        "res": ResNetBuilder,
        "efficientnet": EfficientNetBuilder,
        "effinet": EfficientNetBuilder,
    }

    @classmethod
    def register(cls, name: str, builder_cls: Type[BaseModelBuilder]):
        """Dynamically register a new model architecture class."""
        cls._registry[name.lower()] = builder_cls

    @classmethod
    def create_model(
        cls,
        architecture_name: str = "cnn",
        learning_rate: float = None,
        image_size: Tuple[int, int] = IMAGE_SIZE,
        label_smoothing: float = LABEL_SMOOTHING,
        config: Dict = None
    ) -> Tuple[tf.keras.Model, str]:
        """
        Builds and compiles model instance based on architecture name string.
        Returns tuple of (compiled_model, color_mode).
        """
        arch = architecture_name.lower()
        if arch not in cls._registry:
            valid = list(cls._registry.keys())
            raise ValueError(f"Unknown architecture '{architecture_name}'. Supported: {valid}")

        color_mode = "grayscale" if arch in ["cnn", "custom_cnn"] else "rgb"
        channels = 1 if color_mode == "grayscale" else 3
        input_shape = (image_size[0], image_size[1], channels)

        # Set default learning rates
        if learning_rate is None:
            default_lr = 0.0001 if color_mode == "grayscale" else 0.00001
        else:
            default_lr = learning_rate

        builder_cls = cls._registry[arch]
        builder = builder_cls(input_shape=input_shape, config=config or {})
        model = builder.build()

        optimizer = Adam(learning_rate=default_lr, clipnorm=1.0)
        loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
        )

        return model, color_mode
