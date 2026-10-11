"""Track continuity and quality diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Iterable

import numpy as np


def build_track_diagnostics(
    records: Iterable[Any],
    *,
    frame_width: int,
    frame_height: int,
    fps: float,
    short_track_max_detections: int = 4,
) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[Any]] = defaultdict(list)
    for record in records:
        grouped[record.detection.track_id].append(record)
    diagnostics: dict[int, dict[str, Any]] = {}
    frame_area = float(frame_width * frame_height)
    for track_id, track_records in grouped.items():
        ordered = sorted(track_records, key=lambda item: item.frame_index)
        frames = sorted({item.frame_index for item in ordered})
        first_frame, last_frame = frames[0], frames[-1]
        span = last_frame - first_frame + 1
        gaps = [right - left for left, right in zip(frames, frames[1:])]
        boxes = np.array(
            [[item.detection.x1, item.detection.y1, item.detection.x2, item.detection.y2] for item in ordered],
            dtype=float,
        )
        centres_x = (boxes[:, 0] + boxes[:, 2]) / (2 * frame_width)
        centres_y = (boxes[:, 1] + boxes[:, 3]) / (2 * frame_height)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / frame_area
        confidences = [item.detection.confidence for item in ordered]
        classes = Counter(item.detection.class_name for item in ordered)
        border_contacts = np.mean(
            (boxes[:, 0] <= frame_width * 0.02)
            | (boxes[:, 1] <= frame_height * 0.02)
            | (boxes[:, 2] >= frame_width * 0.98)
            | (boxes[:, 3] >= frame_height * 0.98)
        )
        diagnostics[track_id] = {
            "track_id": track_id,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "duration_frames": span,
            "duration_seconds": span / fps,
            "detection_count": len(ordered),
            "continuity": len(frames) / span,
            "max_gap_frames": max(gaps, default=0),
            "mean_confidence": mean(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "dominant_class": classes.most_common(1)[0][0],
            "class_counts": dict(classes),
            "mean_area_ratio": float(np.mean(areas)),
            "area_coefficient_of_variation": float(np.std(areas) / max(np.mean(areas), 1e-9)),
            "center_x_std": float(np.std(centres_x)),
            "center_y_std": float(np.std(centres_y)),
            "border_contact_ratio": float(border_contacts),
            "selected_count": sum(bool(item.selected) for item in ordered),
            "short_track": len(ordered) <= short_track_max_detections,
        }
    return diagnostics


def diagnostics_summary(diagnostics: dict[int, dict[str, Any]]) -> dict[str, Any]:
    durations = [item["duration_frames"] for item in diagnostics.values()]
    return {
        "track_count": len(diagnostics),
        "short_track_count": sum(bool(item["short_track"]) for item in diagnostics.values()),
        "median_track_duration_frames": median(durations) if durations else 0,
    }
