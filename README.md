# VehicleAI

VehicleAI is a local, offline-first road-vehicle detection and dataset collection pipeline for computer vision research. Version 0.1 uses a pretrained Ultralytics YOLO model to detect generic COCO `car`, `bus`, and `truck` objects in images, videos, and USB camera streams. The `v0.2-classification` branch adds local dataset curation and leakage-resistant dataset preparation; it does not train a classifier yet.

> **Important:** VehicleAI v0.1 detects generic road vehicles. It does **not** currently identify specific vehicle models. It does **not** identify or confirm law-enforcement vehicles.

## Version control and releases

`main` contains stable accepted releases, while `develop` is the integration
branch. New work uses `feature/*` or `fix/*` branches and reviewed pull
requests. Formal versions use annotated semantic-version tags and manually
approved GitHub Releases only after tests pass. See [CHANGELOG.md](CHANGELOG.md)
and [docs/RELEASING.md](docs/RELEASING.md).

The exact v0.1.0 snapshot is preserved by tag `v0.1.0`. The v0.2.0 and v0.2.1
milestones were developed in one uncommitted working tree, so no misleading
historical tags are created for them. Their functionality and recovery status
are recorded in the changelog. Version v0.3.0 is the first exact, tested
post-v0.1 snapshot.

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

## Automated web candidate collection

The collection pipeline uses the documented Wikimedia Commons MediaWiki API.
It stores only candidate images: a search query or collection folder is never
treated as a verified ground-truth label. The default invocation is a local
dry-run with zero network requests and zero downloads:

```powershell
python scripts/collect_web_candidates.py --dry-run
```

After reviewing the provider, queries, counts, and output paths, probe a small
amount of API metadata without downloading image bytes:

```powershell
python scripts/collect_web_candidates.py --probe --probe-limit 1
```

Actual collection is intentionally gated and must be explicitly approved:

```powershell
python scripts/collect_web_candidates.py --download --approve-provider wikimedia_commons
```

Use `--max-per-class 2` for a deliberately small download test after approval.
The configured full target is approximately 100 Charger, 100 Explorer/PIU, 100
Durango, and 200 hard-negative candidates under `data/raw/web/`. Configuration,
queries, quality thresholds, request delay, and target counts live in
`configs/web_collection.yaml`.

Downloaded filenames use
`web_<collection_hint>_<first-12-sha256>.<decoded-format>`. Exact SHA256
duplicates are skipped, pHash near-duplicates are flagged in `metadata.jsonl`,
and corrupted, unsupported, oversized, or low-resolution downloads are
rejected. Metadata retains the provider, query, original and download URLs,
source page, SHA256, pHash, dimensions, author/credit, and available license
fields. Human review changes annotation metadata, never the physical filename.

Collection is sequential, uses a descriptive User-Agent, follows server
rate-limit responses with backoff, and does not bypass authentication, CAPTCHA,
robots restrictions, or anti-scraping controls. The imagery and its metadata
remain ignored by Git. Review each source page and license metadata before any
downstream use.

To review collected web candidates with the existing annotation tool:

```powershell
python scripts/annotate_candidates.py --candidates data/raw/web
```

### Local Chinese web reviewer

For a faster browser-based review workflow, start the local Chinese interface
from the project root:

```powershell
python scripts/review_dataset.py
```

Then open `http://127.0.0.1:8000`. The server binds only to the local loopback
address. It automatically discovers candidates under `data/raw/web/`,
`data/raw/video_candidates/`, and `data/candidates/` when those directories
exist. Use keys `1`–`6` to assign a label, `S` to skip, and the left/right arrow
keys to navigate. Click the displayed image to open its full resolution.

Every label is immediately appended to
`data/annotations/annotations.jsonl`. Relabeling adds a new audit entry while
the latest entry becomes the active label. Refreshing the page or restarting
the server resumes from the remaining unreviewed candidates. Images are never
moved, renamed, or changed. Both candidate data and annotations remain ignored
by Git.

## v0.2: track vehicles and select representative frames

Use YOLO plus ByteTrack to assign a persistent `track_id` to each detected
vehicle and automatically retain at most five useful crops per track:

```powershell
python scripts/track_video.py data/raw/drive.mp4
```

Optional preview and annotated-video output:

```powershell
python scripts/track_video.py data/raw/drive.mp4 --show --save-annotated-video
```

Useful options include `--device cpu`, `--frame-stride 2`, `--max-frames 300`,
`--output-root data/candidates/tracked`, and `--run-name my_session`. Output
directories are required to be new: the command refuses to overwrite an
existing run, existing candidates, or annotations. Tracking always processes
every video frame so ByteTrack motion state stays continuous; `--frame-stride`
only changes how often a frame is considered for candidate selection.

Selection settings live in `configs/tracking.yaml`. Each crop is evaluated for
pixel size, area within the frame, Laplacian sharpness, YOLO confidence, and
visual difference from crops already retained for the same track. Very small or
blurry crops are rejected; near-duplicates are discarded or replace a weaker
duplicate. The per-track limit defaults to five.

VehicleAI uses `configs/bytetrack_vehicleai.yaml` rather than the Ultralytics
default tracker file. YOLO tracking inference runs at confidence 0.10 so
ByteTrack can use low-score detections for its second association stage, while
new tracks still require confidence 0.35. Class-agnostic NMS suppresses the
same physical vehicle being emitted simultaneously as both `car` and `truck`.
These settings affect only `track_video.py`; the stable v0.1 detection scripts
retain their original behavior.

After tracking, lightweight cross-fragment association compares non-overlapping
tracks by time gap, end/start position, scale, vehicle class, and HSV appearance.
Strong evidence creates a `vehicle_group_id`; weaker evidence is recorded as
`suspected_duplicate` for human review. Tracks whose frame ranges overlap are
never automatically grouped.

Candidate quality gating combines track length, continuity, confidence,
geometry, and an optional 5.5 MB ImageNet classifier. The classifier is only
supporting evidence because unusual vehicles and interior objects can confuse
it. Low confidence or a short track alone never causes deletion. Results are
`accepted`, `suspected_false_positive`, or `filtered`; every crop and diagnostic
record is retained. If the classifier is missing or fails, the configured
`metadata_only` fallback keeps uncertain candidates for review.

Each run contains:

- `track_XXXXXX/*.jpg`: selected crops only;
- `metadata.jsonl`: selected samples with video timestamp, frame, track ID,
  detection class/confidence, bounding box, quality metrics, and crop path;
- `selection_log.jsonl`: every evaluated detection, including rejection or
  replacement reason;
- `track_diagnostics.json`: duration, continuity, confidence, class, motion,
  quality status, and verifier evidence for every original Track ID;
- `associations.jsonl`: automatic and suspected cross-track associations;
- `summary.json`: counts by outcome, short tracks, vehicle groups, inferred ID
  switches, and candidate quality status;
- `tracked.mp4`: only when `--save-annotated-video` is requested.

Selected crops are compatible with the existing reviewer because they are
stored under `data/candidates/` with a normal `metadata.jsonl`. Metadata defaults
to `assigned_class=unknown` and `review_status=unreviewed`; the reviewer still
uses the separate append-only annotation database as the authority, so these
defaults do not mark a sample as reviewed.

The web reviewer hides `filtered` candidates by default but exposes explicit
quality filters for accepted, suspected, filtered, or all samples. It displays
the original Track ID, effective vehicle group, all Track IDs in that group,
and suspected duplicates. Reviewers can confirm a merge or split the current
track; these decisions append to `data/annotations/track_groups.jsonl` without
rewriting tracking metadata. The API rejects manual merges when known track
frame ranges overlap.

## v0.3: AI-assisted vehicle-body pre-labeling

Version 0.3 adds an offline-first pre-labeling assistant. It uses the official
DINOv2 ViT-S/14 backbone to extract 384-dimensional, L2-normalized visual
embeddings, then ranks the four body classes with cosine similarity and a
class-balanced kNN score. A score is a similarity value, **not a probability**.
The pre-labeler does not train or claim to be a final real-time classifier.

First audit the current human labels. Only existing images with a current human
four-class label are eligible. `unknown`, `unusable`, filtered candidates,
missing images, conflicting duplicates, and AI-only predictions are excluded:

```powershell
python scripts/audit_prelabel_data.py
```

Download the official small DINOv2 model once and build the incremental cache:

```powershell
python scripts/extract_dinov2_embeddings.py --device auto
```

The model repository and weights are cached under `models/torch_hub/`; image
embeddings are content-addressed in `data/embeddings/dinov2_vits14.npz`.
Unchanged images are reused and newly added images are computed incrementally.
After the first successful run, force local-only loading with:

```powershell
python scripts/extract_dinov2_embeddings.py --offline --device auto
```

Run the identity-group-isolated evaluation. Confirmed vehicle groups used for
threshold calibration are separate from the final validation groups, and both
are separate from references:

```powershell
python scripts/evaluate_prelabeler.py --offline --device auto
```

Reports are written to `results/prelabeling/evaluation.json` and
`results/prelabeling/evaluation.md`. Samples without a trustworthy
`vehicle_id`, `vehicle_group_id`, or video track/session identity remain
reference-only; the report explicitly shows unsupported classes instead of
inventing accuracy values.

Generate predictions for currently unreviewed candidates:

```powershell
python scripts/prelabel_candidates.py --offline --device auto
```

Predictions are stored separately in `data/predictions/prelabels.jsonl` and
never overwrite `data/annotations/annotations.jsonl`. They do not enter the
formal training manifest until a person accepts or changes the label in the
reviewer. Start the reviewer normally:

```powershell
python scripts/review_dataset.py
```

The page displays the recommendation, Top-3 class scores, nearest human
reference image, and an explicit accept button. Existing manual labels cannot
be overwritten by that button. Use “AI 不确定优先” to review low-margin and
`unknown` recommendations first.

## v0.2: annotate candidate crops

The annotation tool reads images recursively from `data/candidates/` and reuses
the nearest v0.1 `metadata.jsonl` records when available. Start it from the
project root:

```powershell
python scripts/annotate_candidates.py
```

Controls:

- `1`: `charger_target` — 2015–2023 Dodge Charger LD body appearance
- `2`: `explorer_target` — 2020–2024 Ford Explorer/PIU body appearance
- `3`: `durango_target` — 2021–2025 Dodge Durango body appearance
- `4`: `other`
- `5`: `unknown` for ambiguous samples
- `6`: `unusable` for corrupt, irrelevant, or unusable samples
- `S`: skip without recording an annotation
- `Q`: save and quit

Each label is immediately appended to `data/annotations/annotations.jsonl`, so
an interruption does not lose earlier work. On restart, already annotated
images are omitted. Use `--include-annotated` to review or relabel them; the
latest annotation for an image wins while earlier entries remain as an audit
trail. Candidate images are only read and are never moved, changed, or deleted.

Annotations record the absolute image path, source image/video, frame number,
track ID, detection confidence, assigned class, and a UTC timestamp whenever
those source fields are available. Annotation files and imagery are ignored by
Git to reduce the risk of committing private road footage.

These labels describe visual body configurations only. They must not be used to
claim that a vehicle is a law-enforcement vehicle, infer its operator, or infer
its use.

## v0.2: curated manifest

After human review, build the metadata-only curated layer from the annotation
history:

```powershell
python scripts/build_curated_manifest.py
```

This writes `data/curated/curated_manifest.jsonl` and prints counts for all six
logical categories. Each original candidate appears at most once and uses its
latest human label. The append-only annotation JSONL remains the separate audit
history. Missing SHA256 values are calculated by reading the original file.
Candidate images are never moved, renamed, modified, or copied, and the curated
manifest remains ignored by Git.

To report counts from the existing manifest without rebuilding it:

```powershell
python scripts/build_curated_manifest.py --summary-only
```

The manifest records the original path, final label, collection hint,
source/provider identifiers, optional vehicle/video/frame/track identifiers,
SHA256, and annotation timestamp. A later dataset builder can consume this
manifest for group-safe splitting; this command does not train a model or
materialise image folders.

## v0.2: prepare the classification dataset

Preview the deterministic split without copying anything:

```powershell
python scripts/prepare_classification_dataset.py --dry-run
```

Materialise the four-class dataset under `data/dataset/`:

```powershell
python scripts/prepare_classification_dataset.py
```

The default ratio is 70/15/15. Override it with, for example,
`--ratios 0.8 0.1 0.1`, and use `--seed` to choose another deterministic split.
All frames sharing the same v0.1 `source` are assigned as a single group, so a
recording cannot leak across train, validation, and test. Samples without source
metadata fall back to per-image grouping; curate their metadata before relying
on them as frames from a shared session. `unknown` annotations stay in the
annotation database and are intentionally excluded from the normal four-class
training folders. Materialisation refuses an output tree that already contains
images; use a new or empty output directory so stale files from an older split
cannot leak across subsets.

## Project layout

```text
VehicleAI/
├── configs/                 Detection configuration
├── data/                    Raw, candidates, annotations, and split data
├── models/                  Future exported/local model artifacts
├── results/                 Annotated images, videos, and metadata
├── scripts/                 Detection, annotation, and preparation commands
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
