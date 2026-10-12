"""Independent-directory blind prediction and later trusted-label evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .curation import IMAGE_SUFFIXES
from .data_audit import TRAINING_LABELS, difference_hash, file_sha256, hamming_distance, normalise_path
from .evaluation import classification_metrics
from .prelabel import Prediction, prediction_record, write_predictions


def discover_blind_images(root: Path) -> list[Path]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Blind-test directory not found: {root}")
    return sorted(
        (path.resolve() for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.as_posix().casefold(),
    )


def blind_prediction_record(image_path: Path, prediction: Prediction, *, model_name: str) -> dict[str, Any]:
    return {
        **prediction_record(image_path, prediction, model_name=model_name),
        "dataset_role": "blind_test",
        "ground_truth": None,
        "review_status": "unreviewed",
    }


def write_blind_predictions(records: Iterable[dict[str, Any]], output_path: Path) -> int:
    return write_predictions(records, output_path)


def load_trusted_labels(path: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return labels
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                label = record.get("label") or record.get("assigned_class")
                if label in TRAINING_LABELS and bool(record.get("identity_confirmed")):
                    labels[normalise_path(record["image_path"])] = record
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid trusted label at {path}:{line_number}: {error}") from error
    return labels


def evaluate_blind_records(
    prediction_records: Iterable[dict[str, Any]], trusted_labels: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    truth: list[str] = []
    predicted: list[str] = []
    top3: list[list[str]] = []
    skipped = 0
    for record in prediction_records:
        label_record = trusted_labels.get(normalise_path(record["image_path"]))
        if label_record is None or "ai_recommendation" not in record:
            skipped += 1
            continue
        truth.append(str(label_record.get("label") or label_record.get("assigned_class")))
        predicted.append(str(record["ai_recommendation"]))
        top3.append([str(item["label"]) for item in record.get("top3", [])])
    return {
        "trusted_evaluated_samples": len(truth),
        "skipped_without_confirmed_identity_label": skipped,
        "metrics": classification_metrics(truth, predicted, top3),
    }


def reject_reference_overlaps(
    paths: Iterable[Path],
    reference_hashes: set[str],
    reference_dhashes: set[str] | None = None,
    *,
    near_duplicate_distance: int = 2,
) -> tuple[list[Path], list[dict[str, Any]]]:
    accepted: list[Path] = []
    rejected: list[dict[str, Any]] = []
    for path in paths:
        digest = file_sha256(path)
        dhash = difference_hash(path)
        exact = digest in reference_hashes
        near = any(hamming_distance(dhash, reference) <= near_duplicate_distance for reference in (reference_dhashes or set()))
        if exact or near:
            rejected.append(
                {
                    "image_path": str(path.resolve()),
                    "dataset_role": "blind_test",
                    "prediction_status": "rejected_reference_overlap" if exact else "rejected_near_reference_overlap",
                    "sha256": digest,
                    "dhash": dhash,
                }
            )
        else:
            accepted.append(path)
    return accepted, rejected
