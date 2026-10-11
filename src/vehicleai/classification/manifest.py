"""Build a metadata-only curated dataset manifest from human annotations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .curation import LABELS, AnnotationStore


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def curated_record(annotation: dict[str, Any]) -> dict[str, Any]:
    """Convert the latest annotation for one image into a curated record."""
    path_value = annotation.get("image_path") or annotation.get("original_path")
    if not path_value:
        raise ValueError("Annotation is missing image_path")
    original_path = Path(str(path_value)).resolve()
    if not original_path.is_file():
        raise FileNotFoundError(f"Annotated candidate does not exist: {original_path}")

    label = annotation.get("label") or annotation.get("assigned_class")
    if label not in LABELS:
        raise ValueError(f"Unsupported curated label for {original_path}: {label!r}")
    timestamp = annotation.get("timestamp") or annotation.get("annotation_timestamp")
    if not timestamp:
        raise ValueError(f"Annotation timestamp is missing for {original_path}")

    sha256 = annotation.get("sha256")
    if not _valid_sha256(sha256):
        sha256 = _file_sha256(original_path)
    source_video = annotation.get("source_video") or annotation.get("video_path")
    if not source_video and (
        annotation.get("frame_number") is not None or annotation.get("frame_index") is not None
    ):
        source_video = annotation.get("source")

    return {
        "original_path": str(original_path),
        "final_label": label,
        "class_hint": annotation.get("class_hint"),
        "source": annotation.get("source"),
        "provider": annotation.get("provider"),
        "source_id": annotation.get("source_id"),
        "vehicle_id": annotation.get("vehicle_id"),
        "source_video": source_video,
        "frame_number": annotation.get("frame_number", annotation.get("frame_index")),
        "track_id": annotation.get("track_id"),
        "sha256": str(sha256).lower(),
        "annotation_timestamp": timestamp,
    }


def build_curated_records(annotations: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = [curated_record(annotation) for annotation in annotations]
    return sorted(records, key=lambda record: str(record["original_path"]).casefold())


def write_curated_manifest(records: Iterable[dict[str, Any]], output_path: Path) -> int:
    """Atomically replace the manifest without touching any candidate image."""
    materialised = list(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in materialised:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return len(materialised)


def build_manifest_from_annotations(annotations_path: Path, output_path: Path) -> list[dict[str, Any]]:
    store = AnnotationStore(annotations_path)
    records = build_curated_records(store.latest.values())
    write_curated_manifest(records, output_path)
    return records


def read_curated_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Curated manifest not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid manifest record at {path}:{line_number}: {error}") from error
    return records


def manifest_summary(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record.get("final_label") for record in records)
    return {label: counts[label] for label in LABELS}
