# brain.md

## Project Purpose

This repository implements a binary image-classification pipeline for identifying space debris versus non-debris objects in orbital imagery. The system is built around TensorFlow/Keras and uses an offline-preprocessed SPARK-2022 dataset cache to support faster, lower-I/O training and inference.

The project’s primary users are:

- ML engineers training and evaluating debris classifiers
- Researchers benchmarking model backbones on SPARK-2022
- Operators or developers running inference on single images or prepared datasets

## High-Level Architecture

The codebase is organized into a small, modular pipeline:

1. Dataset ingestion and preprocessing
   - [src/data/loader.py](src/data/loader.py) parses SPARK-2022 metadata and cached manifests.
   - [src/data/preprocessing.py](src/data/preprocessing.py) performs bounding-box cropping, square padding, resizing, and tensor normalization.
   - [scripts/cache_dataset.py](scripts/cache_dataset.py) builds the preprocessed cache used for fast training.

2. Model construction
   - [src/models/factory.py](src/models/factory.py) acts as the registry/Factory entrypoint.
   - [src/models/base.py](src/models/base.py) and builders in [src/models](src/models) define the architecture-specific model construction flow.
   - Supported architectures are `cnn`, `mobilenet`, `resnet`, and `efficientnet`.

3. Training orchestration
   - [train.py](train.py) is the main training CLI.
   - It loads configuration, sets seeds, initializes GPU settings, loads records, builds a model, creates `SparkDataGenerator` batches, computes class weights, launches Phase 1 and Phase 2 training, then evaluates and visualizes metrics.

4. Inference orchestration
   - [predict.py](predict.py) is the main inference CLI.
   - [src/inference/predictor.py](src/inference/predictor.py) loads a trained model, preprocesses an input image, and returns predicted class probabilities.

5. Evaluation and logging
   - [src/evaluation/metrics.py](src/evaluation/metrics.py) and [src/training/callbacks.py](src/training/callbacks.py) handle metrics, plotting, checkpointing, early stopping, and TensorBoard integration.

## Folder Responsibilities

- [configs/](configs/): central configuration files and runtime defaults.
  - [configs/config.py](configs/config.py): dataclass-based config loader and default constants.
  - [configs/base_config.yaml](configs/base_config.yaml): YAML defaults for model, training, and checkpoint settings.

- [src/data/](src/data/): dataset parsing, metadata loading, caching support, and image preprocessing.

- [src/models/](src/models/): model builders and registry factory.

- [src/inference/](src/inference/): runtime prediction engine.

- [src/training/](src/training/): training callbacks and optimization hooks.

- [src/evaluation/](src/evaluation/): metrics and visualization generation.

- [src/utils/](src/utils/): environment setup utilities such as GPU memory growth configuration.

- [saved_models/](saved_models/): trained model artifacts in `.h5` format.

- [plots/](plots/): evaluation charts and TensorBoard logs.

- [SPARK-2022/](SPARK-2022/): raw dataset layout and label CSVs.

- [SPARK-2022-Preprocessed/](SPARK-2022-Preprocessed/): offline cached 224x224 images and cached label manifests used by the training pipeline.

## Technology Stack

Languages:

- Python 3.x

Core libraries:

- TensorFlow 2.10.x / Keras
- OpenCV
- NumPy
- scikit-learn
- Matplotlib / Seaborn
- Pillow
- imagehash
- PyYAML
- tqdm

Runtime assumptions:

- The project is primarily designed for local or containerized execution.
- GPU acceleration is optional; the code gracefully falls back to CPU when no GPU is visible to TensorFlow.
- The repository expects a dataset directory layout matching SPARK-2022 and its preprocessed cache.

## Dependency Graph

The main runtime dependency chain is:

`train.py` -> `configs.AppConfig` -> `setup_gpu()` -> `load_cached_records()` / `load_spark_split()` -> `ModelFactory.create_model()` -> `SparkDataGenerator` -> `get_callbacks()` -> `model.fit()` -> `evaluate_and_plot()`

`predict.py` -> `setup_gpu()` -> `DebrisPredictor()` -> `ModelFactory.create_model()` -> `preprocess_image()` -> `model.predict()`

The codebase relies on a loose service-style composition pattern rather than a framework-based application server. The main “dependency edges” are configuration -> data -> model -> training/evaluation -> artifacts.

## Execution Flow

### Training flow

1. Parse CLI arguments in [train.py](train.py).
2. Load centralized config from [configs/base_config.yaml](configs/base_config.yaml) via [configs/config.py](configs/config.py).
3. Set deterministic seeds and configure TensorFlow GPU memory growth.
4. Load records from either:
   - the preprocessed offline cache in [SPARK-2022-Preprocessed/](SPARK-2022-Preprocessed/), or
   - the raw SPARK-2022 directory if the cache is missing.
5. Instantiate a model using the registry in [src/models/factory.py](src/models/factory.py).
6. Create train/validation/test `SparkDataGenerator` sequence objects.
7. Compute class weights from the training labels.
8. Run Phase 1 warmup for the classification head.
9. If enabled, unfreeze backbone layers and run Phase 2 fine-tuning.
10. Save the best checkpoint, render training curves, and run final evaluation on the test split.

### Inference flow

1. Parse input image path and model type in [predict.py](predict.py).
2. Initialize GPU settings.
3. Create a `DebrisPredictor`.
4. Load model weights or a full `.h5` model file.
5. Preprocess the image using the color mode determined by the model class.
6. Run sigmoid inference and convert the raw probability into debris vs non-debris output.

## Request Lifecycle

There is no web request lifecycle in the usual server sense. The project is a batch/CLI workflow:

- Input: local image path(s), local dataset cache, or raw SPARK-2022 split data
- Processing: preprocessing, batch generation, model training or prediction
- Output: `.h5` model artifacts, metrics plots, and classification predictions

For the current codebase, the “request lifecycle” is effectively the training or prediction invocation itself.

## Database Design

This project does not use a relational database or document database.

Storage model:

- Dataset manifests are lightweight CSV files under [SPARK-2022-Preprocessed/labels/](SPARK-2022-Preprocessed/labels/).
- Preprocessed images are stored on disk as normalized 224x224 image files.
- Model checkpoints are saved to [saved_models/](saved_models/).
- Evaluation artifacts and logs are saved to [plots/](plots/).

Schema notes:

- `cached_{split}.csv` contains at minimum:
  - `cached_path`: path to a preprocessed image file
  - `label`: numeric binary label (`0` = debris, `1` = non-debris)

## API Contracts

This repository does not expose a network API layer. The closest thing to an API contract is the command-line interface.

Primary entrypoint contracts:

- `python train.py --model <cnn|mobilenet|resnet|efficientnet> ...`
- `python predict.py --image <path> --model <weights_or_model_path> --type <cnn|mobilenet|resnet|efficientnet>`

Model contract assumptions:

- All model builders return a compiled Keras `Model`.
- The output layer is a single sigmoid node representing the probability of the non-debris class.
- The classifier interprets the final probability as:
  - `prob_non_debris = model_output`
  - `prob_debris = 1 - prob_non_debris`

## Key Algorithms and Business Logic

### Binary classification objective

The task is framed as a binary classification problem:

- `debris` -> class `0`
- `non_debris` -> class `1`

### Data-label normalization

The SPARK loader maps raw annotation classes to a binary label scheme and intentionally collapses all active-satellite categories into the non-debris class.

### Preprocessing logic

The preprocessing pipeline performs several important steps:

- boundary-safe bounding-box crop
- square padding to preserve object presence without aspect distortion
- resizing to `224x224`
- grayscale or RGB normalization to `[0, 1]`

### Training strategy

The training script uses two phases:

- Phase 1: train only the classifier head
- Phase 2: unfreeze a portion of the backbone and fine-tune the remaining network

This is the main optimization pattern for transfer-learning models such as MobileNet, ResNet, and EfficientNet.

### Imbalance handling

Class weights are computed dynamically from the training label distribution and are passed to `model.fit()` to mitigate the skew between debris and non-debris examples.

## Configuration

Key configuration sources:

- [configs/base_config.yaml](configs/base_config.yaml)
- [configs/config.py](configs/config.py)

Notable defaults:

- image size: `224x224`
- batch size: `32`
- epochs: `30`
- learning rate: `0.0001`
- label smoothing: `0.0` or `0.1` depending on execution context
- seed: `42`

Environment variables are not used as a primary configuration mechanism in this repo. Configuration is file-based and code-driven.

## Environment Variables

The codebase does not define a strong environment-variable contract. The effective runtime environment is primarily determined by:

- local filesystem paths,
- TensorFlow GPU visibility,
- Python path assumptions (`PYTHONPATH=/app` in the Dockerfile), and
- the selected virtual environment or container image.

Unknowns:

- No explicit secret or auth environment schema is present in the current repository.
- No production secrets management pattern is implemented in the codebase.

## Coding Standards

Observed conventions in the repo:

- Python modules use clear docstrings and top-level descriptions.
- CLI entrypoints are implemented as `main()` functions guarded by `if __name__ == "__main__":`.
- Config constants are centralized in [configs/config.py](configs/config.py).
- Model creation is via registry/factory rather than direct import coupling.
- Batch-level image loading is done through `tf.keras.utils.Sequence` generators instead of naive full in-memory loading for large data.

Naming conventions:

- `train.py` and `predict.py` act as orchestrators.
- builders are named by architecture, e.g. `CustomCNNBuilder`, `MobileNetBuilder`, `ResNetBuilder`, `EfficientNetBuilder`.
- binary class names use `debris` and `non_debris` in config and parser logic.

## Reusable Patterns

The codebase clearly uses these reusable design patterns:

- Factory pattern: [src/models/factory.py](src/models/factory.py)
- Builder pattern: [src/models](src/models)
- Sequence-based data loading: [src/data/preprocessing.py](src/data/preprocessing.py)
- Callback composition for training lifecycle management: [src/training/callbacks.py](src/training/callbacks.py)
- Modular package structure with package entrypoints in each `src/*/__init__.py`

## Error Handling

Current patterns:

- Missing model paths raise `FileNotFoundError`.
- Unknown model names raise `ValueError`.
- Dataset cache fallback is handled gracefully in [train.py](train.py).
- Preprocessing handles missing/invalid bounding boxes by falling back to center crop / resize.
- GPU configuration errors are caught and logged; the code continues with a CPU fallback path when possible.

Observed gaps:

- There is no dedicated structured logging framework.
- Error handling is mostly operational and print-driven rather than typed exception hierarchy-based.
- No centralized telemetry or alerting layer is present.

## Security Practices

The current repository does not implement an application security boundary. There is no authentication, authorization, user session layer, or external secret store integration.

What is present:

- no embedded secrets in the checked-in code
- no user-facing network surface
- no data persistence beyond local filesystem artifacts

Security risks to be aware of:

- the training pipeline reads local files and archives from disk
- the repository assumes that input image paths and dataset directories are trusted
- model artifacts and logs are stored in local directories without an access-control layer

## Performance Considerations

The project’s performance strategy is mostly centered on data-path optimization:

- offline dataset preprocessing via [scripts/cache_dataset.py](scripts/cache_dataset.py)
- `SparkDataGenerator` for GPU-friendly batch loading
- dynamic GPU memory growth in [src/utils/gpu.py](src/utils/gpu.py)
- cropping and padding once offline rather than per batch
- image resizing to a fixed `224x224` target

Known performance characteristics:

- caching can dramatically reduce training startup and batch I/O overhead
- transfer-learning models are cheaper and more practical than full end-to-end training from scratch
- the code currently favors deterministic local training over distributed training or multi-worker serving

## External Integrations

The current repository has limited external integrations.

Present integrations:

- TensorFlow and Keras for model training and inference
- OpenCV for image decoding and preprocessing
- optional GPU hardware via TensorFlow
- TensorBoard logs for experiment monitoring

Absent or not implemented in the codebase:

- database backend
- message queue
- object storage
- external auth provider
- CI/CD service integration
- remote model registry
- web API or SDK

## Testing Strategy

The repository does not currently show a dedicated formal test suite in the top-level structure.

What can be inferred:

- validation and test metrics are generated during training via `evaluate_and_plot()`
- the code is expected to be validated by running `train.py` and `predict.py` end-to-end
- the dataset cache script serves as a useful integration check for preprocessing correctness

Known gap:

- there is no visible `tests/` folder, pytest configuration, or CI workflow file in the current tree.

## CI/CD Pipeline

The repository includes a Dockerfile for containerized execution, but no explicit GitHub Actions or GitLab CI configuration is present in the tree snapshot shown here.

Expected manual deployment path:

- build a container image from [Dockerfile](Dockerfile)
- install dependencies from [requirements.txt](requirements.txt)
- execute training or prediction commands inside the container

## Deployment

Current deployment pattern:

- local Python environment or containerized environment
- direct CLI execution of training or inference
- saved model artifact export to [saved_models/](saved_models/)

Container deployment is supported by [Dockerfile](Dockerfile), which uses a TensorFlow GPU image and installs Python requirements.

## Common Commands

Training:

- `python train.py --model cnn --epochs 30 --batch-size 16`
- `python train.py --model mobilenet --epochs 20`
- `python train.py --model resnet --epochs 20`
- `python train.py --model efficientnet --epochs 20`

Inference:

- `python predict.py --image path/to/image.jpg --model saved_models/cnn_spark_debris.h5 --type cnn`

Dataset caching:

- `python scripts/cache_dataset.py --spark-dir SPARK-2022 --target-dir SPARK-2022-Preprocessed`

## Important Files

- [train.py](train.py): main training orchestrator
- [predict.py](predict.py): main inference orchestrator
- [configs/config.py](configs/config.py): central configuration loader
- [src/models/factory.py](src/models/factory.py): model registry and construction entrypoint
- [src/data/preprocessing.py](src/data/preprocessing.py): image preprocessing utilities and batch generator
- [src/inference/predictor.py](src/inference/predictor.py): inference runtime wrapper
- [scripts/cache_dataset.py](scripts/cache_dataset.py): offline cache builder
- [Dockerfile](Dockerfile): container deployment definition

## Known Limitations

The current repository has several explicit and implicit constraints:

- It is a local CLI workflow, not a web service.
- It depends on the presence of SPARK-2022 dataset files and/or the preprocessed cache directory.
- The code is heavily optimized around a fixed `224x224` image size.
- There is no visible production-grade secrets management, observability, or multi-tenant architecture.
- The repository appears to be research/ops oriented rather than a distributed production platform.

## Assumptions and Unknowns

Verified assumptions:

- The model’s final output is a single sigmoid probability used for binary classification.
- The repository is intended to classify images using either custom CNN or pretrained backbones.
- The offline cache is the preferred runtime path for training because it removes repeated crop/resize work.

Unknowns that should be explicitly documented if future work expands the system:

- whether any deployment environment or model registry exists outside the local repo
- whether the training pipeline is expected to run in a remote cluster or in CI
- whether the system will eventually need API authentication, audit trails, or a database-backed experiment metadata store

## Data Flow Overview

Input image or dataset record -> preprocessing -> tensor batch generation -> model forward pass -> binary probability -> class decision -> metric/log/artifact output

This is the core data path that should be preserved when extending the code.

## Maintenance Guidelines

To safely modify this project:

1. Keep configuration centralized in [configs/config.py](configs/config.py) and [configs/base_config.yaml](configs/base_config.yaml).
2. Preserve the `train.py` -> `ModelFactory` -> `SparkDataGenerator` -> `callbacks` workflow unless there is a strong reason to replace it.
3. Avoid duplicating model logic across builders; add architecture-specific behavior only inside the appropriate model builder.
4. Preserve the offline cache workflow if training speed matters, because it is an important part of the repository’s performance profile.
5. When changing model output semantics, update both the training loss logic and inference probability interpretation together.
6. Prefer small, reversible changes that preserve compatibility with existing CLI entrypoints.
7. When adding new features, document them here so this file remains the single source of truth for project understanding.

## Summary

This repository is a compact, production-oriented research pipeline for binary space-debris classification. Its architecture is intentionally simple:

- local filesystem data ingestion
- offline preprocessing cache
- factory-based model construction
- Keras training loop with callbacks
- CLI prediction engine
- artifact generation for evaluation and deployment

The document above is intended to be a durable mental model for future maintenance, code reviews, and onboarding.
