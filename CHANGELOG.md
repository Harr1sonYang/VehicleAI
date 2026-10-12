# Changelog

All notable VehicleAI changes are documented here. Versions follow Semantic Versioning where a recoverable release commit exists.

## [0.3.2] - Unreleased

### Added

- Content-addressed MP4/MOV/AVI/MKV/WebM inbox scanning with a durable SQLite processing queue.
- Queue progress/status reporting, bounded automatic retry, interruption recovery, and stable per-video session IDs.
- A one-command road-video pipeline that reuses YOLO, ByteTrack, representative-frame selection, quality filtering, and cached DINOv2 pre-labeling.
- Cross-video exact and perceptual near-duplicate indexing that records evidence without asserting vehicle identity.
- Collection statistics for videos, queue status, tracks, candidates, human labels, and strict-validation identities.
- High-value reviewer ordering that prioritizes predicted target vehicles while retaining Other and Unknown candidates.
- Tests for queue recovery, content deduplication, collection statistics, batch metadata, and review priority.
- Responsive reviewer preview controls for zoom, fit-to-window, native crop size, and opening the original video frame.
- Independent `detection_valid` and `classification_usable` metadata with conservative fine-grained classification quality reasons and filtering.

### Changed

- Reworked the Flask reviewer into a large preview workspace and wider sticky review sidebar with labels and navigation kept near the image.
- Moved paths, hashes, Track IDs, vehicle identity history, and related provenance into a collapsed advanced-information panel.
- Replaced raw confusion-matrix JSON with a table and moved model evaluation and error cases into collapsed panels.
- Candidate crops below 128×96 pixels remain available for distance/detection research but are marked `needs_review` instead of being treated as fine-grained classification-ready.

### Fixed

- Small crops such as 75×58 candidates are now visibly enlarged for review with an explicit warning that CSS scaling does not add image detail.
- Legacy candidates without an explicit classification-quality record are evaluated conservatively from their actual pixel dimensions.

### Safety

- Source videos are never moved, modified, or deleted.
- Failed partial outputs are preserved outside the review candidate root before retry.
- Content similarity and cross-video duplication never automatically merge real-world vehicle identities.
- AI predictions remain separate from append-only human annotations.

### Known limitations

- The MVP processes one queued video at a time; it does not run concurrent GPU workers.
- Queue progress is stage-level. Per-frame progress remains visible in the active process rather than persisted for every decoded frame.

## [0.3.1] - 2026-10-12

### Added

- Append-only, evidence-aware `vehicle_identity` records independent from Track ID and automatic track groups.
- Manual vehicle identity merge, split, unknown, and suspected-duplicate actions in the Flask reviewer.
- Strict Reference/Calibration/Test splitting with identity and duplicate leakage protection.
- Per-class independent-vehicle counts, `insufficient_data` status, and explicit data-shortage reporting.
- Independent-directory DINOv2+kNN blind testing with Top-3, abstention, margin, nearest reference, and JSONL output.
- Optional blind-test metrics that require later human labels with confirmed vehicle identity.
- Cross-source DINOv2 similarity suggestions that never automatically merge identities.
- Reviewer display of identity members, suspected sources, confusion matrix, and error cases.

### Changed

- Automatic cross-track `vehicle_group_id` association is no longer treated as verified real-world identity.
- Test samples are never used for threshold selection; thresholds come from the separate Calibration set.

### Known limitations

- Charger, Explorer, and Durango currently have zero confirmed independent vehicle identities and remain `insufficient_data` for strict evaluation.

## [0.3.0] - 2026-10-11

### Added

- DINOv2 ViT-S/14 CUDA/CPU embedding extraction with offline, incremental caching.
- Class-balanced cosine kNN pre-labeling with Top-3 results, abstention, and nearest-reference evidence.
- Identity-group-isolated calibration and validation reports.
- AI recommendations, uncertainty sorting, and explicit recommendation acceptance in the Flask reviewer.
- Tests covering audit safety, embedding caching, balanced prediction, evaluation isolation, and reviewer integration.
- Long-term `main`/`develop`/`feature/*`/`fix/*` branch and release governance.
- CPU GitHub Actions for pushes and pull requests, plus separately marked manual CUDA checks.
- A test-gated, manually initiated GitHub Release workflow.

### Safety

- AI predictions are stored separately and cannot overwrite existing human annotations.
- Raw data, annotations, embeddings, predictions, results, and model weights remain excluded from Git.

### Known limitations

- The target-class images do not yet have independently verified vehicle identities, so target-class accuracy cannot be claimed.
- This is the first exact post-v0.1 Git snapshot. The v0.2.0 and v0.2.1 milestones were developed in the same uncommitted working tree and cannot be reconstructed as honest standalone commits.

## v0.2.1 historical milestone - no recoverable release commit

- Added cross-track association, track diagnostics, candidate-quality filtering, and lightweight false-positive verification.
- No tag is created because an exact v0.2.1 file snapshot is not available.

## v0.2.0 historical milestone - no recoverable release commit

- Added YOLO + ByteTrack video tracking and representative-frame selection.
- Added the local curation/review workflow and leakage-resistant dataset preparation.
- No tag is created because an exact v0.2.0 file snapshot is not available.

## [0.1.0] - 2026-10-09

- Initial vehicle detection and crop pipeline.
- Recoverable commit: `50c1ab8`.
- Annotated Git tag: `v0.1.0`.
