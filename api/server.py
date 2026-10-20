"""
FastAPI Inference Server for the Space Debris Identification System.

Exposes the trained Keras checkpoints over HTTP so the Next.js frontend (frontend/)
can classify uploaded imagery with any of the 4 architectures and render Grad-CAM audits.

Usage (from repository root):
    python -m uvicorn api.server:app --port 8000
"""
import base64
import json
import os
import sys
import tempfile
import threading
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.utils import setup_gpu
from src.inference import DebrisPredictor
from src.data.preprocessing import preprocess_image
from src.evaluation.gradcam import find_last_conv_layer, make_gradcam_heatmap

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
METRICS_JSON = os.path.join(ROOT_DIR, "plots", "comparison", "model_comparison_metrics.json")
SAMPLE_DIRS = {
    "debris": os.path.join(ROOT_DIR, "sample_debris"),
    "non_debris": os.path.join(ROOT_DIR, "sample_non_debris"),
}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

MODEL_KEYS = ["cnn", "mobilenet", "efficientnet", "resnet"]

# Fallback copy of plots/comparison/model_comparison_metrics.json (plots/ is gitignored)
FALLBACK_METRICS = {
    "cnn": {"Architecture": "Custom CNN", "Total Parameters": 422785, "Trainable Parameters": 421825,
            "Checkpoint Size (MB)": 1.65, "GPU Latency (ms/img)": 1.42, "Best Val Loss": 0.0375,
            "Peak Val Accuracy (%)": 99.72, "Peak Val Precision (%)": 99.95,
            "Peak Val Recall (%)": 99.89, "Peak Val F1-Score (%)": 99.92},
    "mobilenet": {"Architecture": "MobileNetV2", "Total Parameters": 2427713, "Trainable Parameters": 1677633,
                  "Checkpoint Size (MB)": 9.49, "GPU Latency (ms/img)": 2.51, "Best Val Loss": 0.1277,
                  "Peak Val Accuracy (%)": 99.78, "Peak Val Precision (%)": 99.95,
                  "Peak Val Recall (%)": 99.95, "Peak Val F1-Score (%)": 99.95},
    "resnet": {"Architecture": "ResNet-50", "Total Parameters": 23858817, "Trainable Parameters": 15220225,
               "Checkpoint Size (MB)": 91.25, "GPU Latency (ms/img)": 4.05, "Best Val Loss": 0.1215,
               "Peak Val Accuracy (%)": 99.89, "Peak Val Precision (%)": 99.99,
               "Peak Val Recall (%)": 99.95, "Peak Val F1-Score (%)": 99.97},
    "efficientnet": {"Architecture": "EfficientNet-B0", "Total Parameters": 4219300, "Trainable Parameters": 3296797,
                     "Checkpoint Size (MB)": 16.36, "GPU Latency (ms/img)": 3.36, "Best Val Loss": 0.1232,
                     "Peak Val Accuracy (%)": 99.9, "Peak Val Precision (%)": 99.99,
                     "Peak Val Recall (%)": 99.97, "Peak Val F1-Score (%)": 99.98},
}

setup_gpu()

app = FastAPI(title="SpaceGuard Vision API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

for _cls, _dir in SAMPLE_DIRS.items():
    if os.path.isdir(_dir):
        app.mount(f"/samples/{_cls}", StaticFiles(directory=_dir), name=f"samples_{_cls}")

_predictors = {}
_load_lock = threading.Lock()
_infer_lock = threading.Lock()


def _weights_path(model_key: str) -> str:
    return os.path.join(SAVED_MODELS_DIR, f"{model_key}_spark_debris.h5")


def _load_metrics() -> dict:
    try:
        with open(METRICS_JSON, "r", encoding="utf-8") as f:
            return {row["Model Key"]: row for row in json.load(f)}
    except Exception:
        return FALLBACK_METRICS


def _get_predictor(model_key: str) -> DebrisPredictor:
    if model_key not in MODEL_KEYS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_key}'. Choose from {MODEL_KEYS}.")
    if model_key in _predictors:
        return _predictors[model_key]
    path = _weights_path(model_key)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Weights for '{model_key}' not found at {path}.")
    with _load_lock:
        if model_key not in _predictors:
            _predictors[model_key] = DebrisPredictor(model_path=path, model_type=model_key)
    return _predictors[model_key]


def _png_base64(img_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _render_gradcam(predictor: DebrisPredictor, image_path: str) -> dict:
    """Builds input / heatmap / overlay images, blended like run_zero_trust_audit()."""
    img_tensor = preprocess_image(image_path, color_mode=predictor.color_mode, model_type=predictor.model_type)
    backbone_name, last_conv_name = find_last_conv_layer(predictor.model)
    heatmap = make_gradcam_heatmap(img_tensor, predictor.model, backbone_name, last_conv_name)

    img_rgb = np.repeat(img_tensor, 3, axis=-1) if img_tensor.shape[-1] == 1 else img_tensor.copy()
    img_display = (img_rgb / 255.0 if np.max(img_rgb) > 1.0 else img_rgb).astype(np.float32)

    heatmap_resized = cv2.resize(heatmap, (img_display.shape[1], img_display.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = (cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
    overlay = cv2.addWeighted(img_display, 0.6, heatmap_color, 0.4, 0)

    to_u8 = lambda a: np.uint8(np.clip(a, 0.0, 1.0) * 255)
    return {
        "layer": last_conv_name,
        "input": _png_base64(to_u8(img_display)),
        "heatmap": _png_base64(to_u8(heatmap_color)),
        "overlay": _png_base64(to_u8(overlay)),
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "loaded_models": sorted(_predictors.keys()),
        "gpu": bool(tf.config.list_physical_devices("GPU")),
    }


@app.get("/api/models")
def list_models():
    metrics = _load_metrics()
    models = []
    for key in MODEL_KEYS:
        m = metrics.get(key, FALLBACK_METRICS[key])
        models.append({
            "key": key,
            "name": m["Architecture"],
            "total_params": m["Total Parameters"],
            "trainable_params": m["Trainable Parameters"],
            "size_mb": m["Checkpoint Size (MB)"],
            "latency_ms": m["GPU Latency (ms/img)"],
            "val_loss": m["Best Val Loss"],
            "accuracy": m["Peak Val Accuracy (%)"],
            "precision": m["Peak Val Precision (%)"],
            "recall": m["Peak Val Recall (%)"],
            "f1": m["Peak Val F1-Score (%)"],
            "available": os.path.exists(_weights_path(key)),
            "loaded": key in _predictors,
        })
    return {"models": models}


@app.get("/api/samples")
def list_samples():
    samples = []
    for cls, directory in SAMPLE_DIRS.items():
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if name.lower().endswith(IMAGE_EXTENSIONS):
                samples.append({"name": name, "label": cls, "url": f"/samples/{cls}/{name}"})
    return {"samples": samples}


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form("cnn"),
    threshold: float = Form(0.5),
    allow_uncertain: bool = Form(True),
    gradcam: bool = Form(True),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    if not 0.0 < threshold < 1.0:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1.")
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 15 MB upload limit.")

    model_key = model.lower()
    predictor = _get_predictor(model_key)

    suffix = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        if cv2.imread(tmp_path) is None:
            raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

        with _infer_lock:
            start = time.perf_counter()
            result = predictor.predict(image_path=tmp_path, threshold=threshold, allow_uncertain=allow_uncertain)
            inference_ms = (time.perf_counter() - start) * 1000.0

            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])

            cam, cam_warning = None, None
            if gradcam:
                try:
                    cam = _render_gradcam(predictor, tmp_path)
                except Exception as e:
                    cam_warning = f"Grad-CAM unavailable: {e}"
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    result.pop("image_path", None)
    return {
        **result,
        "model": model_key,
        "filename": file.filename,
        "inference_ms": round(inference_ms, 2),
        "gradcam": cam,
        "warning": cam_warning,
    }
