import numpy as np

from vehicleai.temporal.frame_selection import SelectionRecord
from vehicleai.tracking.association import associate_track_fragments
from vehicleai.tracking.tracker import TrackedDetection


def record(track_id: int, frame: int, image, *, x: int = 20, class_name: str = "car") -> SelectionRecord:
    item = SelectionRecord(
        source="drive.mp4",
        frame_index=frame,
        timestamp_ms=frame * 33.3,
        detection=TrackedDetection(track_id, 2, class_name, 0.8, x, 20, x + 80, 80),
        crop=image.copy(),
        quality_score=0.8,
        selected=True,
    )
    return item


def textured(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (60, 80, 3), dtype=np.uint8)


def test_same_vehicle_reappearing_after_track_break_is_grouped() -> None:
    image = textured(1)
    records = [record(1, frame, image, x=20 + frame) for frame in range(3)]
    records += [record(2, frame, image, x=24 + frame) for frame in range(6, 9)]
    result = associate_track_fragments(records, frame_width=320, frame_height=180)
    assert result.group_by_track[1] == result.group_by_track[2]
    assert result.status_by_track[1] == "auto_merged"


def test_tracks_in_same_frame_are_never_grouped_even_if_identical() -> None:
    image = textured(2)
    records = [record(1, frame, image, x=20) for frame in range(3)]
    records += [record(2, frame, image, x=22) for frame in range(3)]
    result = associate_track_fragments(records, frame_width=320, frame_height=180)
    assert result.group_by_track[1] != result.group_by_track[2]
    assert result.associations == []


def test_visually_different_sequential_vehicles_are_not_merged() -> None:
    first = np.zeros((60, 80, 3), dtype=np.uint8)
    first[:, :] = (0, 0, 255)
    second = np.zeros((60, 80, 3), dtype=np.uint8)
    second[:, :] = (0, 255, 0)
    records = [record(1, frame, first) for frame in range(3)]
    records += [record(2, frame, second) for frame in range(6, 9)]
    result = associate_track_fragments(records, frame_width=320, frame_height=180)
    assert result.group_by_track[1] != result.group_by_track[2]


def test_ambiguous_sequential_match_is_only_suspected() -> None:
    image = textured(8)
    records = [record(1, frame, image, x=20) for frame in range(3)]
    records += [record(2, frame, image, x=75) for frame in range(6, 9)]
    result = associate_track_fragments(records, frame_width=320, frame_height=180)
    assert result.group_by_track[1] != result.group_by_track[2]
    assert result.status_by_track[1] == "suspected_duplicate"
    assert result.suspected_by_track[1] == [2]


def test_transitive_merge_cannot_create_overlapping_vehicle_group() -> None:
    image = textured(3)
    records = [record(1, frame, image) for frame in range(0, 4)]
    records += [record(2, frame, image) for frame in range(7, 10)]
    records += [record(3, frame, image) for frame in range(3, 6)]
    result = associate_track_fragments(records, frame_width=320, frame_height=180)
    groups = result.group_members.values()
    assert all(not ({1, 3} <= set(group)) for group in groups)
