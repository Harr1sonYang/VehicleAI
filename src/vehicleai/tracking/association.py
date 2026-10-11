"""Conservative cross-fragment association without a heavyweight ReID model."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class Association:
    earlier_track_id: int
    later_track_id: int
    frame_gap: int
    score: float
    appearance_similarity: float
    spatial_similarity: float
    scale_similarity: float
    class_similarity: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class AssociationResult:
    group_by_track: dict[int, str]
    group_members: dict[str, list[int]]
    status_by_track: dict[int, str]
    suspected_by_track: dict[int, list[int]]
    associations: list[Association]


def _histogram(image: Any) -> Any:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    return cv2.normalize(histogram, histogram).flatten()


def _appearance_similarity(first_records: list[Any], second_records: list[Any]) -> float:
    first = sorted(first_records, key=lambda item: item.quality_score, reverse=True)[:3]
    second = sorted(second_records, key=lambda item: item.quality_score, reverse=True)[:3]
    scores = [
        (cv2.compareHist(_histogram(left.crop), _histogram(right.crop), cv2.HISTCMP_CORREL) + 1.0) / 2.0
        for left in first
        for right in second
    ]
    return float(max(scores, default=0.0))


def associate_track_fragments(
    records: Iterable[Any],
    *,
    frame_width: int,
    frame_height: int,
    max_gap_frames: int = 90,
    max_spatial_distance: float = 0.30,
    min_scale_similarity: float = 0.35,
    suspected_score: float = 0.68,
    auto_merge_score: float = 0.86,
    min_auto_appearance: float = 0.72,
) -> AssociationResult:
    grouped: dict[int, list[Any]] = defaultdict(list)
    for record in records:
        grouped[record.detection.track_id].append(record)
    for track_records in grouped.values():
        track_records.sort(key=lambda item: item.frame_index)

    candidates: list[Association] = []
    track_ids = sorted(grouped)
    diagonal = float(np.hypot(frame_width, frame_height))
    compatible_vehicle_classes = {"car", "truck"}
    for first_index, first_id in enumerate(track_ids):
        for second_id in track_ids[first_index + 1 :]:
            first_records, second_records = grouped[first_id], grouped[second_id]
            if first_records[-1].frame_index < second_records[0].frame_index:
                earlier_id, later_id = first_id, second_id
                earlier, later = first_records, second_records
            elif second_records[-1].frame_index < first_records[0].frame_index:
                earlier_id, later_id = second_id, first_id
                earlier, later = second_records, first_records
            else:
                # Any temporal overlap means the tracks could be two vehicles.
                continue
            frame_gap = later[0].frame_index - earlier[-1].frame_index - 1
            if frame_gap > max_gap_frames:
                continue
            left_box = earlier[-1].detection
            right_box = later[0].detection
            left_center = np.array([(left_box.x1 + left_box.x2) / 2, (left_box.y1 + left_box.y2) / 2])
            right_center = np.array([(right_box.x1 + right_box.x2) / 2, (right_box.y1 + right_box.y2) / 2])
            distance = float(np.linalg.norm(left_center - right_center) / diagonal)
            if distance > max_spatial_distance:
                continue
            spatial_similarity = 1.0 - distance / max_spatial_distance
            left_area = max(1, (left_box.x2 - left_box.x1) * (left_box.y2 - left_box.y1))
            right_area = max(1, (right_box.x2 - right_box.x1) * (right_box.y2 - right_box.y1))
            scale_similarity = min(left_area, right_area) / max(left_area, right_area)
            if scale_similarity < min_scale_similarity:
                continue
            left_class = Counter(item.detection.class_name for item in earlier).most_common(1)[0][0]
            right_class = Counter(item.detection.class_name for item in later).most_common(1)[0][0]
            if left_class == right_class:
                class_similarity = 1.0
            elif {left_class, right_class}.issubset(compatible_vehicle_classes):
                class_similarity = 0.75
            else:
                class_similarity = 0.25
            appearance = _appearance_similarity(earlier, later)
            score = 0.40 * appearance + 0.30 * spatial_similarity + 0.15 * scale_similarity + 0.15 * class_similarity
            if score >= auto_merge_score and appearance >= min_auto_appearance:
                status = "auto_merged"
            elif score >= suspected_score:
                status = "suspected_duplicate"
            else:
                continue
            candidates.append(
                Association(
                    earlier_id,
                    later_id,
                    frame_gap,
                    score,
                    appearance,
                    spatial_similarity,
                    scale_similarity,
                    class_similarity,
                    status,
                )
            )

    parent = {track_id: track_id for track_id in track_ids}
    frames_by_root = {track_id: {item.frame_index for item in grouped[track_id]} for track_id in track_ids}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for association in sorted(candidates, key=lambda item: item.score, reverse=True):
        if association.status != "auto_merged":
            continue
        left_root = find(association.earlier_track_id)
        right_root = find(association.later_track_id)
        if left_root == right_root or frames_by_root[left_root] & frames_by_root[right_root]:
            continue
        keep, merge = min(left_root, right_root), max(left_root, right_root)
        parent[merge] = keep
        frames_by_root[keep] |= frames_by_root[merge]

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for track_id in track_ids:
        members_by_root[find(track_id)].append(track_id)
    group_by_track: dict[int, str] = {}
    group_members: dict[str, list[int]] = {}
    for root, members in sorted(members_by_root.items()):
        group_id = f"vehicle_group_{min(members):06d}"
        group_members[group_id] = sorted(members)
        for track_id in members:
            group_by_track[track_id] = group_id

    suspected: dict[int, set[int]] = defaultdict(set)
    for association in candidates:
        if association.status == "suspected_duplicate":
            suspected[association.earlier_track_id].add(association.later_track_id)
            suspected[association.later_track_id].add(association.earlier_track_id)
    status_by_track = {
        track_id: (
            "auto_merged"
            if len(group_members[group_by_track[track_id]]) > 1
            else "suspected_duplicate"
            if suspected[track_id]
            else "single"
        )
        for track_id in track_ids
    }
    return AssociationResult(
        group_by_track,
        group_members,
        status_by_track,
        {track_id: sorted(values) for track_id, values in suspected.items()},
        sorted(candidates, key=lambda item: (-item.score, item.earlier_track_id, item.later_track_id)),
    )
