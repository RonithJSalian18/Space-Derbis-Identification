import numpy as np
import pytest
from src.data.preprocessing import apply_architecture_preprocessing, SparkDataGenerator

def test_apply_architecture_preprocessing():
    # Synthetic RGB image in [0, 255]
    img = np.ones((224, 224, 3), dtype=np.uint8) * 128

    # CNN expects [0.0, 1.0]
    out_cnn = apply_architecture_preprocessing(img, model_type="cnn")
    assert 0.0 <= out_cnn.min() and out_cnn.max() <= 1.0
    assert out_cnn.max() == pytest.approx(128.0 / 255.0, abs=1e-3)

    # Pretrained models expect standard [0, 255] float32 arrays
    out_res = apply_architecture_preprocessing(img, model_type="resnet")
    assert out_res.dtype == np.float32

    out_eff = apply_architecture_preprocessing(img, model_type="efficientnet")
    assert out_eff.dtype == np.float32

def test_generator_label_alignment(tmp_path):
    import cv2
    img_path = str(tmp_path / "sample.jpg")
    cv2.imwrite(img_path, np.zeros((224, 224, 3), dtype=np.uint8))

    records = [
        {"path": img_path, "label": 0},
        {"path": img_path, "label": 1},
        {"path": img_path, "label": 0}
    ]

    gen = SparkDataGenerator(records, batch_size=2, shuffle=False)
    assert len(gen.labels) == len(records)
    assert np.array_equal(gen.labels, np.array([0, 1, 0]))

    # First batch
    X_b0, y_b0 = gen[0]
    assert len(X_b0) == 2
    assert np.array_equal(y_b0, np.array([0, 1]))
