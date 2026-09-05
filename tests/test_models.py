import pytest
import tensorflow as tf
from src.models import ModelFactory, unfreeze_backbone


def test_model_factory_compilation_metrics():
    model, color_mode = ModelFactory.create_model('cnn')
    metric_names = [m.name if hasattr(m, 'name') else str(m) for m in model.compiled_metrics._metrics]
    assert 'pr_auc' in metric_names
    assert 'roc_auc' in metric_names
    assert 'precision' in metric_names
    assert 'recall' in metric_names


def test_unfreeze_backbone_mobilenet():
    model, _ = ModelFactory.create_model('mobilenet')
    arch = unfreeze_backbone(model, 'mobilenet', fine_tune_blocks=2)
    assert arch == 'mobilenet'

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            assert layer.trainable is False
        if isinstance(layer, tf.keras.Model):
            for sub_l in layer.layers:
                if isinstance(sub_l, tf.keras.layers.BatchNormalization):
                    assert sub_l.trainable is False


def test_unfreeze_backbone_resnet():
    model, _ = ModelFactory.create_model('resnet')
    arch = unfreeze_backbone(model, 'resnet', fine_tune_stage='conv5')
    assert arch == 'resnet'

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            assert layer.trainable is False
        if isinstance(layer, tf.keras.Model):
            for sub_l in layer.layers:
                if isinstance(sub_l, tf.keras.layers.BatchNormalization):
                    assert sub_l.trainable is False
