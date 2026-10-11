"""Class-balanced cosine kNN predictions kept separate from human labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .data_audit import TRAINING_LABELS, normalise_path
from .embeddings import l2_normalise


@dataclass(frozen=True)
class Prediction:
    predicted_label: str
    similarity: float
    margin: float
    top3: tuple[dict[str, Any], ...]
    nearest_reference_path: str | None


class BalancedKNNClassifier:
    """Score each class from its own top-k neighbours, preventing majority voting."""

    def __init__(self, *, k_per_class: int = 3, similarity_threshold: float = 0.55, margin_threshold: float = 0.02) -> None:
        if k_per_class < 1:
            raise ValueError("k_per_class must be positive")
        self.k_per_class = k_per_class
        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold

    def fit(self, embeddings: np.ndarray, labels: Iterable[str], paths: Iterable[str | Path]) -> "BalancedKNNClassifier":
        self.embeddings = l2_normalise(embeddings)
        self.labels = np.asarray(list(labels))
        self.paths = np.asarray([str(Path(path).resolve()) for path in paths])
        if len(self.embeddings) != len(self.labels) or len(self.labels) != len(self.paths):
            raise ValueError("Reference embeddings, labels, and paths must have equal lengths")
        if not len(self.labels):
            raise ValueError("At least one reference is required")
        return self

    def predict_one(self, embedding: np.ndarray) -> Prediction:
        query = l2_normalise(np.asarray(embedding, dtype=np.float32).reshape(1, -1))[0]
        similarities = self.embeddings @ query
        rankings: list[dict[str, Any]] = []
        for label in TRAINING_LABELS:
            indices = np.flatnonzero(self.labels == label)
            if not len(indices):
                continue
            ordered = indices[np.argsort(similarities[indices])[::-1]]
            neighbours = ordered[: self.k_per_class]
            nearest = int(ordered[0])
            rankings.append(
                {
                    "label": label,
                    "score": float(np.mean(similarities[neighbours])),
                    "nearest_similarity": float(similarities[nearest]),
                    "nearest_reference_path": str(self.paths[nearest]),
                }
            )
        rankings.sort(key=lambda item: item["score"], reverse=True)
        if not rankings:
            return Prediction("unknown", 0.0, 0.0, (), None)
        best = rankings[0]
        second_score = rankings[1]["score"] if len(rankings) > 1 else -1.0
        margin = float(best["score"] - second_score)
        label = best["label"]
        if best["score"] < self.similarity_threshold or margin < self.margin_threshold:
            label = "unknown"
        return Prediction(
            predicted_label=label,
            similarity=float(best["score"]),
            margin=margin,
            top3=tuple(rankings[:3]),
            nearest_reference_path=best["nearest_reference_path"],
        )

    def predict(self, embeddings: np.ndarray) -> list[Prediction]:
        return [self.predict_one(row) for row in embeddings]


def prediction_record(image_path: Path, prediction: Prediction, *, model_name: str) -> dict[str, Any]:
    return {
        "image_path": str(image_path.resolve()),
        "ai_recommendation": prediction.predicted_label,
        "similarity": prediction.similarity,
        "margin": prediction.margin,
        "top3": list(prediction.top3),
        "nearest_reference_path": prediction.nearest_reference_path,
        "model_name": model_name,
        "score_semantics": "cosine similarity; not a probability",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def write_predictions(records: Iterable[dict[str, Any]], output_path: Path) -> int:
    materialised = list(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in materialised:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(materialised)


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                latest[normalise_path(record["image_path"])] = record
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid prediction at {path}:{line_number}: {error}") from error
    return latest
