"""Online representative-frame selection for tracked vehicle crops."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from vehicleai.tracking.tracker import TrackedDetection


DEFAULT_CLASSIFICATION_MIN_WIDTH = 128
DEFAULT_CLASSIFICATION_MIN_HEIGHT = 96
DEFAULT_CLASSIFICATION_MIN_AREA_RATIO = 0.0025
DEFAULT_CLASSIFICATION_MIN_SHARPNESS = 35.0


def assess_classification_usability(
    *,
    width: int,
    height: int,
    area_ratio: float | None,
    sharpness: float | None,
    min_width: int = DEFAULT_CLASSIFICATION_MIN_WIDTH,
    min_height: int = DEFAULT_CLASSIFICATION_MIN_HEIGHT,
    min_area_ratio: float = DEFAULT_CLASSIFICATION_MIN_AREA_RATIO,
    min_sharpness: float = DEFAULT_CLASSIFICATION_MIN_SHARPNESS,
) -> dict[str, Any]:
    """Assess fine-grained classification detail independently from detection validity."""
    reasons: list[str] = []
    if width < min_width or height < min_height:
        reasons.append("insufficient_pixel_dimensions")
    if area_ratio is not None and area_ratio < min_area_ratio:
        reasons.append("insufficient_frame_area")
    if sharpness is not None and sharpness < min_sharpness:
        reasons.append("insufficient_detail_sharpness")

    resolution_score = min(1.0, width / max(min_width, 1), height / max(min_height, 1))
    area_score = 1.0 if area_ratio is None else min(1.0, area_ratio / max(min_area_ratio, 1e-9))
    sharpness_score = 1.0 if sharpness is None else min(1.0, sharpness / max(min_sharpness, 1e-9))
    score = 0.60 * resolution_score + 0.25 * area_score + 0.15 * sharpness_score
    usable = not reasons
    return {
        "classification_usable": usable,
        "classification_quality_status": "usable" if usable else "needs_review",
        "classification_quality_score": round(float(score), 6),
        "classification_quality_reasons": reasons,
    }


@dataclass
class SelectionRecord:
    source: str
    frame_index: int
    timestamp_ms: float
    detection: TrackedDetection
    crop: Any = field(repr=False)
    sharpness: float = 0.0
    area_ratio: float = 0.0
    quality_score: float = 0.0
    nearest_difference: float | None = None
    selected: bool = False
    filter_reason: str | None = None
    crop_path: str | None = None
    vehicle_group_id: str | None = None
    group_track_ids: list[int] = field(default_factory=list)
    association_status: str = "single"
    suspected_duplicate_track_ids: list[int] = field(default_factory=list)
    candidate_status: str = "accepted"
    candidate_filter_reasons: list[str] = field(default_factory=list)
    track_statistics: dict[str, Any] = field(default_factory=dict)
    secondary_verification: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    source_video_id: str | None = None
    detection_valid: bool = True
    classification_usable: bool = False
    classification_quality_status: str = "needs_review"
    classification_quality_score: float = 0.0
    classification_quality_reasons: list[str] = field(default_factory=list)
    _thumbnail: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_video": self.source,
            "session_id": self.session_id,
            "source_video_id": self.source_video_id,
            "timestamp_ms": self.timestamp_ms,
            "timestamp_seconds": self.timestamp_ms / 1000.0,
            "frame_index": self.frame_index,
            "frame_number": self.frame_index,
            "track_id": self.detection.track_id,
            "class_id": self.detection.class_id,
            "class_name": self.detection.class_name,
            "confidence": self.detection.confidence,
            "detection_confidence": self.detection.confidence,
            "bbox_xyxy": [
                self.detection.x1,
                self.detection.y1,
                self.detection.x2,
                self.detection.y2,
            ],
            "sharpness": self.sharpness,
            "area_ratio": self.area_ratio,
            "quality_score": self.quality_score,
            "nearest_frame_difference": self.nearest_difference,
            "selected": self.selected,
            "filter_reason": self.filter_reason,
            "crop_path": self.crop_path,
            "vehicle_group_id": self.vehicle_group_id,
            "group_track_ids": self.group_track_ids,
            "association_status": self.association_status,
            "suspected_duplicate_track_ids": self.suspected_duplicate_track_ids,
            "candidate_status": self.candidate_status,
            "candidate_filter_reasons": self.candidate_filter_reasons,
            "track_statistics": self.track_statistics,
            "secondary_verification": self.secondary_verification,
            "detection_valid": self.detection_valid,
            "classification_usable": self.classification_usable,
            "classification_quality_status": self.classification_quality_status,
            "classification_quality_score": self.classification_quality_score,
            "classification_quality_reasons": self.classification_quality_reasons,
            "review_status": "unreviewed",
            "assigned_class": "unknown",
            "label": "unknown",
        }


def crop_tracked_detection(frame: Any, detection: TrackedDetection, padding: float = 0.0) -> Any:
    height, width = frame.shape[:2]
    box_width = detection.x2 - detection.x1
    box_height = detection.y2 - detection.y1
    pad_x = int(box_width * padding)
    pad_y = int(box_height * padding)
    x1 = max(0, detection.x1 - pad_x)
    y1 = max(0, detection.y1 - pad_y)
    x2 = min(width, detection.x2 + pad_x)
    y2 = min(height, detection.y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Tracked detection produced an empty crop")
    return frame[y1:y2, x1:x2].copy()


def image_sharpness(crop: Any) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_difference(first: Any, second: Any, size: int = 64) -> float:
    first_gray = cv2.resize(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), (size, size))
    second_gray = cv2.resize(cv2.cvtColor(second, cv2.COLOR_BGR2GRAY), (size, size))
    return float(np.mean(cv2.absdiff(first_gray, second_gray)) / 255.0)


class FrameSelector:
    """Keep a small, sharp, large, confident, and visually diverse set per track."""

    def __init__(
        self,
        *,
        max_per_track: int = 5,
        min_width: int = 64,
        min_height: int = 48,
        min_area_ratio: float = 0.001,
        min_sharpness: float = 35.0,
        min_difference: float = 0.06,
        thumbnail_size: int = 64,
        classification_min_width: int = DEFAULT_CLASSIFICATION_MIN_WIDTH,
        classification_min_height: int = DEFAULT_CLASSIFICATION_MIN_HEIGHT,
        classification_min_area_ratio: float = DEFAULT_CLASSIFICATION_MIN_AREA_RATIO,
        classification_min_sharpness: float = DEFAULT_CLASSIFICATION_MIN_SHARPNESS,
    ) -> None:
        if max_per_track < 1:
            raise ValueError("max_per_track must be at least 1")
        self.max_per_track = max_per_track
        self.min_width = min_width
        self.min_height = min_height
        self.min_area_ratio = min_area_ratio
        self.min_sharpness = min_sharpness
        self.min_difference = min_difference
        self.thumbnail_size = thumbnail_size
        self.classification_min_width = classification_min_width
        self.classification_min_height = classification_min_height
        self.classification_min_area_ratio = classification_min_area_ratio
        self.classification_min_sharpness = classification_min_sharpness
        self.records: list[SelectionRecord] = []
        self._selected: dict[int, list[SelectionRecord]] = {}
        self._track_ids: set[int] = set()

    def consider(
        self,
        frame: Any,
        detection: TrackedDetection,
        *,
        source: str,
        frame_index: int,
        timestamp_ms: float,
        padding: float = 0.0,
        session_id: str | None = None,
        source_video_id: str | None = None,
    ) -> SelectionRecord:
        crop = crop_tracked_detection(frame, detection, padding)
        crop_height, crop_width = crop.shape[:2]
        frame_height, frame_width = frame.shape[:2]
        area_ratio = (crop_width * crop_height) / float(frame_width * frame_height)
        sharpness = image_sharpness(crop)
        size_score = min(1.0, area_ratio / 0.10)
        sharpness_score = min(1.0, sharpness / 500.0)
        quality_score = 0.50 * detection.confidence + 0.25 * size_score + 0.25 * sharpness_score
        classification_quality = assess_classification_usability(
            width=crop_width,
            height=crop_height,
            area_ratio=area_ratio,
            sharpness=sharpness,
            min_width=self.classification_min_width,
            min_height=self.classification_min_height,
            min_area_ratio=self.classification_min_area_ratio,
            min_sharpness=self.classification_min_sharpness,
        )
        record = SelectionRecord(
            source=source,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            detection=detection,
            crop=crop,
            sharpness=sharpness,
            area_ratio=area_ratio,
            quality_score=quality_score,
            session_id=session_id,
            source_video_id=source_video_id,
            **classification_quality,
            _thumbnail=cv2.resize(crop, (self.thumbnail_size, self.thumbnail_size)),
        )
        self.records.append(record)
        self._track_ids.add(detection.track_id)

        if crop_width < self.min_width or crop_height < self.min_height or area_ratio < self.min_area_ratio:
            record.filter_reason = "too_small"
            return record
        if sharpness < self.min_sharpness:
            record.filter_reason = "too_blurry"
            return record

        selected = self._selected.setdefault(detection.track_id, [])
        if selected:
            differences = [frame_difference(record._thumbnail, item._thumbnail, self.thumbnail_size) for item in selected]
            nearest_index = int(np.argmin(differences))
            record.nearest_difference = differences[nearest_index]
            if record.nearest_difference < self.min_difference:
                similar = selected[nearest_index]
                if record.quality_score > similar.quality_score:
                    similar.selected = False
                    similar.filter_reason = "replaced_by_better_duplicate"
                    selected[nearest_index] = record
                    record.selected = True
                else:
                    record.filter_reason = "duplicate"
                return record

        if len(selected) < self.max_per_track:
            record.selected = True
            selected.append(record)
            return record

        weakest = min(selected, key=lambda item: item.quality_score)
        if record.quality_score > weakest.quality_score:
            weakest.selected = False
            weakest.filter_reason = "replaced_by_higher_quality"
            selected[selected.index(weakest)] = record
            record.selected = True
        else:
            record.filter_reason = "selection_limit"
        return record

    def selected_records(self) -> list[SelectionRecord]:
        return sorted(
            (record for records in self._selected.values() for record in records if record.selected),
            key=lambda item: (item.detection.track_id, item.frame_index),
        )

    def summary(self) -> dict[str, Any]:
        reasons = Counter(record.filter_reason or "selected" for record in self.records)
        return {
            "evaluated_detections": len(self.records),
            "selected_images": len(self.selected_records()),
            "track_count": len(self._track_ids),
            "selected_track_count": sum(bool(records) for records in self._selected.values()),
            "outcomes": dict(sorted(reasons.items())),
        }
