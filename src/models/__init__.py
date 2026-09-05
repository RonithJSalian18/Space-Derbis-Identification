import tensorflow as tf
from .base import BaseModelBuilder
from .factory import ModelFactory
from .builder import get_model
from .cnn import build_custom_cnn, CustomCNNBuilder
from .mobilenet import build_mobilenet, MobileNetBuilder, unfreeze_mobilenet
from .resnet import build_resnet, ResNetBuilder, unfreeze_resnet
from .efficientnet import build_efficientnet, EfficientNetBuilder, unfreeze_efficientnet


def unfreeze_backbone(model, arch_name: str = None, **kwargs):
    """
    Unified architecture-aware backbone unfreezer.
    Locks Batch Normalization layers in inference mode while selectively unfreezing top blocks.
    """
    arch = (arch_name or "").lower()
    if not arch:
        layer_names = [l.name.lower() for l in model.layers]
        if any("resnet" in n for n in layer_names):
            arch = "resnet"
        elif any("efficientnet" in n for n in layer_names):
            arch = "efficientnet"
        elif any("mobilenet" in n for n in layer_names):
            arch = "mobilenet"

    result = None
    if "resnet" in arch:
        stage = kwargs.get("fine_tune_stage", "conv5")
        unfreeze_resnet(model, fine_tune_stage=stage)
        result = "resnet"
    elif "efficientnet" in arch or "effinet" in arch:
        blocks = kwargs.get("fine_tune_blocks", 2)
        unfreeze_efficientnet(model, fine_tune_blocks=blocks)
        result = "efficientnet"
    elif "mobilenet" in arch or "mobile" in arch:
        blocks = kwargs.get("fine_tune_blocks", 2)
        unfreeze_mobilenet(model, fine_tune_blocks=blocks)
        result = "mobilenet"

    # Deep locking of all BatchNormalization layers across the entire model graph
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.BatchNormalization,)):
            layer.trainable = False
        if isinstance(layer, tf.keras.Model):
            for sub_l in layer.layers:
                if isinstance(sub_l, (tf.keras.layers.BatchNormalization,)):
                    sub_l.trainable = False

    return result


__all__ = [
    "BaseModelBuilder",
    "ModelFactory",
    "get_model",
    "build_custom_cnn",
    "CustomCNNBuilder",
    "build_mobilenet",
    "MobileNetBuilder",
    "unfreeze_mobilenet",
    "build_resnet",
    "ResNetBuilder",
    "unfreeze_resnet",
    "build_efficientnet",
    "EfficientNetBuilder",
    "unfreeze_efficientnet",
    "unfreeze_backbone",
]

