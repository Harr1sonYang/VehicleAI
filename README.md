# VehicleAI

VehicleAI is a local, offline-first road-vehicle detection and dataset collection pipeline for computer vision research. Version 0.1 uses a pretrained Ultralytics YOLO model to detect generic COCO `car`, `bus`, and `truck` objects in images, videos, and USB camera streams.

> **Important:** VehicleAI v0.1 detects generic road vehicles. It does **not** currently identify specific vehicle models. It does **not** identify or confirm law-enforcement vehicles.

## Current scope

- Image detection with annotated output and JSON metadata
- Video-file and USB-camera detection with annotated MP4 and JSONL metadata
- Vehicle crop extraction from images, image directories, videos, or cameras
- Automatic CUDA selection with CPU fallback
- A small YAML configuration file at `configs/detection.yaml`

## Windows setup

Install Python 3.10 or 3.11, clone the repository, and run these commands from the `VehicleAI` project root in PowerShell:

```powershell
py -3.11 -m venv .venv

# Activate the environment:
.\.venv\Scripts\Activate.ps1

# Upgrade packaging tools:
python -m pip install --upgrade pip setuptools wheel

# Install an NVIDIA CUDA build of PyTorch (recommended for the RTX 4060):
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install VehicleAI and the remaining dependencies:
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The NVIDIA driver may report a newer CUDA capability than the PyTorch wheel. That is expected: the driver is backward compatible with the CUDA runtime bundled in PyTorch. A separate CUDA Toolkit installation is not required for this v0.1 pipeline.

The first detection run downloads the small pretrained `yolo11n.pt` weights into `models/`. Model weights are intentionally not stored in this repository. Later inference is local and does not use a cloud vision API.

## Verify the installation

```powershell
python scripts/test_gpu.py
python -m pytest
```

`test_gpu.py` returns exit code 1 when CUDA is unavailable so setup problems are visible. Detection itself still falls back to CPU.

## Detect vehicles in an image

```powershell
python scripts/detect_image.py data/raw/road.jpg --save-crops
```

Outputs are written to `results/images/`. Override this with `--output-dir`. Use `--device cpu` to force CPU inference.

## Detect vehicles in video

```powershell
python scripts/detect_video.py data/raw/drive.mp4
python scripts/detect_video.py data/raw/drive.mp4 --show
```

Use `--frame-stride 2` to infer every second frame. Frames skipped by inference remain unannotated in the saved video.

## Use a USB camera

Camera indices normally begin at zero:

```powershell
python scripts/detect_video.py 0 --show --no-save
```

Press `q` in the preview window to stop. Try camera index `1` if another camera is selected first.

## Build a crop collection

```powershell
# One image or all supported images directly inside a directory
python scripts/extract_vehicle_crops.py data/raw

# Video: sample every fifth frame by default
python scripts/extract_vehicle_crops.py data/raw/drive.mp4 --frame-stride 5

# USB camera
python scripts/extract_vehicle_crops.py 0 --frame-stride 5 --max-frames 300
```

Crops and append-only `metadata.jsonl` are written to `data/candidates/`. Keep original videos grouped by recording session and vehicle identity; later train/validation/test splitting must be group-based rather than random frame-based to avoid leakage.

## Configuration

Edit `configs/detection.yaml` to change the YOLO model, confidence threshold, IoU threshold, inference size, allowed COCO class IDs, or crop padding. `yolo11n.pt` is intentionally small and is a good working baseline for an 8 GB laptop GPU.

## Project layout

```text
VehicleAI/
├── configs/                 Detection configuration
├── data/                    Raw, candidate, processed, and split data
├── models/                  Future exported/local model artifacts
├── results/                 Annotated images, videos, and metadata
├── scripts/                 Runnable v0.1 commands
├── src/vehicleai/           Reusable pipeline code
└── tests/                   Fast utility tests
```

## Practical limitations

COCO vehicle detection is not vehicle-model or rear-view classification. Detection quality depends on target pixel size, exposure, focus, motion blur, weather, and camera lens/FOV. The saved crops are candidates for manual curation, not automatically valid training examples. A generic vehicle detection must never be interpreted as identifying a specific trim, package, owner, operator, or use.

This project does not include datasets, road footage, vehicle crops, generated results, pretrained weights, or virtual environments. Users are responsible for collecting and handling imagery lawfully and protecting privacy, including visible faces and license plates.

## Planned development

Future versions may add:

- Curated full-rear vehicle classification with a strong `other` class
- Explicit `unknown`/reject behavior for poor or ambiguous imagery
- Vehicle tracking and multi-frame temporal aggregation
- A separate taillight-focused research model
- Independent vehicle color classification
- Evaluation by distance, angle, lighting, and false alerts per hour
- ONNX/TensorRT export and later edge-device evaluation

These items are roadmap goals, not capabilities or performance claims of v0.1.
