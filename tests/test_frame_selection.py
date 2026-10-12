import numpy as np

from vehicleai.temporal.frame_selection import FrameSelector, frame_difference
from vehicleai.tracking.tracker import TrackedDetection


def detection(track_id: int = 1, confidence: float = 0.9, size: int = 100) -> TrackedDetection:
    return TrackedDetection(track_id, 2, "car", confidence, 0, 0, size, size)


def textured_frame(seed: int, size: int = 120) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


def test_small_and_blurry_frames_are_filtered() -> None:
    selector = FrameSelector(min_width=50, min_height=50, min_sharpness=20)
    small = selector.consider(
        textured_frame(1), detection(size=20), source="video.mp4", frame_index=0, timestamp_ms=0
    )
    blurry_frame = np.full((120, 120, 3), 127, dtype=np.uint8)
    blurry = selector.consider(
        blurry_frame, detection(size=100), source="video.mp4", frame_index=1, timestamp_ms=40
    )
    assert small.filter_reason == "too_small"
    assert blurry.filter_reason == "too_blurry"


def test_better_duplicate_replaces_previous_frame() -> None:
    selector = FrameSelector(min_sharpness=0, min_difference=0.1)
    frame = textured_frame(2)
    first = selector.consider(
        frame, detection(confidence=0.5), source="video.mp4", frame_index=0, timestamp_ms=0
    )
    second = selector.consider(
        frame.copy(), detection(confidence=0.95), source="video.mp4", frame_index=1, timestamp_ms=40
    )
    assert frame_difference(first.crop, second.crop) == 0.0
    assert not first.selected and first.filter_reason == "replaced_by_better_duplicate"
    assert second.selected
    assert selector.selected_records() == [second]


def test_selection_limit_is_per_track() -> None:
    selector = FrameSelector(max_per_track=2, min_sharpness=0, min_difference=0)
    records = [
        selector.consider(
            textured_frame(seed),
            detection(confidence=0.7 + seed / 100),
            source="video.mp4",
            frame_index=seed,
            timestamp_ms=seed * 40,
        )
        for seed in range(4)
    ]
    assert len(selector.selected_records()) == 2
    assert sum(record.selected for record in records) == 2
    assert selector.summary()["evaluated_detections"] == 4


def test_metadata_defaults_to_unknown_and_unreviewed() -> None:
    selector = FrameSelector(min_sharpness=0)
    record = selector.consider(
        textured_frame(9), detection(track_id=7), source="drive.mp4", frame_index=3, timestamp_ms=120
    )
    metadata = record.to_dict()
    assert metadata["track_id"] == 7
    assert metadata["bbox_xyxy"] == [0, 0, 100, 100]
    assert metadata["assigned_class"] == "unknown"
    assert metadata["review_status"] == "unreviewed"


def test_metadata_preserves_batch_session_and_video_identity() -> None:
    selector = FrameSelector(min_sharpness=0)
    record = selector.consider(
        textured_frame(10),
        detection(track_id=8),
        source="drive.mov",
        frame_index=4,
        timestamp_ms=160,
        session_id="drive-a1b2c3",
        source_video_id="a" * 64,
    )
    metadata = record.to_dict()
    assert metadata["session_id"] == "drive-a1b2c3"
    assert metadata["source_video_id"] == "a" * 64
    assert metadata["quality_score"] > 0
