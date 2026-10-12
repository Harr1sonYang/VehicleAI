"""Candidate discovery, v0.1 metadata lookup, and durable annotations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABELS = ("charger_target", "explorer_target", "durango_target", "other", "unknown", "unusable")
KEY_LABELS = {
    ord("1"): "charger_target",
    ord("2"): "explorer_target",
    ord("3"): "durango_target",
    ord("4"): "other",
    ord("5"): "unknown",
    ord("6"): "unusable",
}


def _normalise_path(path: Path) -> str:
    return path.resolve().as_posix().casefold()


@dataclass(frozen=True)
class Candidate:
    image_path: Path
    source: str | None = None
    frame_number: int | None = None
    track_id: int | str | None = None
    detection_confidence: float | None = None
    class_hint: str | None = None
    provider: str | None = None
    query: str | None = None
    source_url: str | None = None
    source_page_url: str | None = None
    sha256: str | None = None
    phash: str | None = None
    width: int | None = None
    height: int | None = None
    source_id: str | int | None = None
    vehicle_id: str | int | None = None
    source_video: str | None = None
    vehicle_group_id: str | None = None
    group_track_ids: tuple[int, ...] = ()
    association_status: str | None = None
    suspected_duplicate_track_ids: tuple[int, ...] = ()
    candidate_status: str | None = None
    candidate_filter_reasons: tuple[str, ...] = ()
    track_first_frame: int | None = None
    track_last_frame: int | None = None
    session_id: str | None = None
    source_video_id: str | None = None
    timestamp_ms: float | None = None
    quality_score: float | None = None

    @property
    def key(self) -> str:
        return _normalise_path(self.image_path)


class AnnotationStore:
    """Append-only JSONL annotation store; the latest entry wins."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._latest = self._load_latest()

    def _load_latest(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        if not self.path.exists():
            return latest
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    latest[_normalise_path(Path(record["image_path"]))] = record
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError(f"Invalid annotation at {self.path}:{line_number}: {error}") from error
        return latest

    @property
    def latest(self) -> dict[str, dict[str, Any]]:
        return dict(self._latest)

    def is_annotated(self, candidate: Candidate) -> bool:
        return candidate.key in self._latest

    def record(
        self,
        candidate: Candidate,
        assigned_class: str,
        *,
        annotation_method: str = "manual",
    ) -> dict[str, Any]:
        if assigned_class not in LABELS:
            raise ValueError(f"Unsupported class: {assigned_class}")
        record = {
            **asdict(candidate),
            "image_path": str(candidate.image_path.resolve()),
            "assigned_class": assigned_class,
            "label": assigned_class,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "annotation_method": annotation_method,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        self._latest[candidate.key] = record
        return record


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _metadata_index(root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for metadata_path in root.rglob("metadata.jsonl"):
        for record in _read_jsonl(metadata_path):
            crop_value = record.get("crop_path") or record.get("image_path") or record.get("filename")
            if not crop_value:
                continue
            crop_path = Path(str(crop_value))
            candidates = [crop_path]
            if not crop_path.is_absolute():
                candidates.extend((metadata_path.parent / crop_path, Path.cwd() / crop_path))
            for candidate_path in candidates:
                index[_normalise_path(candidate_path)] = record
                index[candidate_path.name.casefold()] = record
    return index


def discover_candidates(root: Path) -> list[Candidate]:
    """Find candidate images and enrich them with available v0.1 metadata."""
    if not root.is_dir():
        raise FileNotFoundError(f"Candidate directory not found: {root}")
    metadata = _metadata_index(root)
    result: list[Candidate] = []
    for image_path in sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
        record = metadata.get(_normalise_path(image_path), metadata.get(image_path.name.casefold(), {}))
        track_statistics = record.get("track_statistics") or {}
        result.append(
            Candidate(
                image_path=image_path.resolve(),
                source=record.get("source") or record.get("source_page_url") or record.get("source_url"),
                frame_number=record.get("frame_index", record.get("frame_number")),
                track_id=record.get("track_id"),
                detection_confidence=record.get("confidence", record.get("detection_confidence")),
                class_hint=record.get("class_hint"),
                provider=record.get("provider"),
                query=record.get("query"),
                source_url=record.get("source_url"),
                source_page_url=record.get("source_page_url"),
                sha256=record.get("sha256"),
                phash=record.get("phash"),
                width=record.get("width"),
                height=record.get("height"),
                source_id=record.get("source_id", record.get("provider_page_id")),
                vehicle_id=record.get("vehicle_id"),
                source_video=(
                    record.get("source_video")
                    or record.get("video_path")
                    or (
                        record.get("source")
                        if record.get("frame_index") is not None or record.get("frame_number") is not None
                        else None
                    )
                ),
                vehicle_group_id=record.get("vehicle_group_id"),
                group_track_ids=tuple(record.get("group_track_ids") or ()),
                association_status=record.get("association_status"),
                suspected_duplicate_track_ids=tuple(record.get("suspected_duplicate_track_ids") or ()),
                candidate_status=record.get("candidate_status"),
                candidate_filter_reasons=tuple(record.get("candidate_filter_reasons") or ()),
                track_first_frame=track_statistics.get("first_frame"),
                track_last_frame=track_statistics.get("last_frame"),
                session_id=record.get("session_id"),
                source_video_id=record.get("source_video_id"),
                timestamp_ms=record.get("timestamp_ms"),
                quality_score=record.get("quality_score"),
            )
        )
    return result
