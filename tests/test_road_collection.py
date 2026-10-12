import json
from pathlib import Path

from PIL import Image

from vehicleai.collection.road import (
    VideoQueue,
    collection_statistics,
    find_cross_video_duplicates,
)
from vehicleai.review import create_app


def _candidate(root: Path, session: str, name: str, source: str, color=(20, 40, 60)) -> Path:
    directory = root / session / "track_000001"
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / name
    Image.new("RGB", (80, 60), color=color).save(image_path)
    record = {
        "crop_path": str(image_path.resolve()),
        "source": source,
        "source_video": source,
        "session_id": session,
        "source_video_id": f"hash-{session}",
        "timestamp_ms": 1250.0,
        "frame_index": 25,
        "track_id": 1,
        "vehicle_group_id": "vehicle_group_000001",
        "quality_score": 0.81,
        "candidate_status": "accepted",
    }
    (root / session / "metadata.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return image_path


def test_video_queue_discovers_content_once_and_recovers_interruption(tmp_path: Path) -> None:
    inbox = tmp_path / "incoming"
    inbox.mkdir()
    (inbox / "drive.MP4").write_bytes(b"same-video-content")
    (inbox / "renamed.mov").write_bytes(b"same-video-content")
    (inbox / "notes.txt").write_text("ignored", encoding="utf-8")
    queue = VideoQueue(tmp_path / "queue.sqlite3")

    discovery = queue.discover(inbox)
    assert discovery == {"found": 2, "added": 1, "duplicate_content": 1}
    assert queue.discover(inbox)["added"] == 0
    item = queue.next()
    assert item is not None and item["attempts"] == 1
    queue.update(item["video_id"], status="tracking", stage="yolo_bytetrack")
    assert queue.recover_interrupted() == 1
    resumed = queue.next()
    assert resumed is not None and resumed["video_id"] == item["video_id"]
    assert resumed["attempts"] == 2
    queue.update(resumed["video_id"], status="completed", stage="done")
    assert queue.next() is None


def test_cross_video_duplicates_are_flagged_without_identity_merge(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    first = _candidate(root, "session-a", "one.jpg", "a.mp4")
    second = _candidate(root, "session-b", "two.jpg", "b.mp4")
    near = _candidate(root, "session-c", "three.jpg", "c.mp4", color=(90, 100, 110))
    second.write_bytes(first.read_bytes())

    records = find_cross_video_duplicates(root)
    by_path = {record["image_path"]: record for record in records}
    assert by_path[str(first.resolve())]["duplicate_status"] == "unique"
    duplicate = by_path[str(second.resolve())]
    assert duplicate["duplicate_status"] == "exact"
    assert duplicate["duplicate_of"] == str(first.resolve())
    assert duplicate["identity_action"] == "none"
    assert by_path[str(near.resolve())]["duplicate_status"] == "near"
    assert by_path[str(near.resolve())]["identity_action"] == "none"


def test_collection_statistics_include_tracks_labels_and_confirmed_identities(tmp_path: Path) -> None:
    inbox = tmp_path / "incoming"
    inbox.mkdir()
    video = inbox / "drive.mp4"
    video.write_bytes(b"video")
    queue = VideoQueue(tmp_path / "queue.sqlite3")
    queue.discover(inbox)
    item = queue.next()
    assert item is not None
    root = tmp_path / "candidates"
    output = root / item["session_id"]
    image = _candidate(root, item["session_id"], "car.jpg", str(video.resolve()))
    (output / "summary.json").write_text(
        json.dumps({"track_count": 3, "selected_images": 7}), encoding="utf-8"
    )
    queue.update(item["video_id"], status="completed", stage="done", output_dir=output)
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(
        json.dumps({"image_path": str(image.resolve()), "label": "charger_target"}) + "\n",
        encoding="utf-8",
    )

    statistics = collection_statistics(
        queue, root, annotations, tmp_path / "vehicle_identities.jsonl"
    )
    assert statistics["queue"]["video_count"] == 1
    assert statistics["independent_tracks"] == 3
    assert statistics["candidate_images"] == 7
    assert statistics["human_confirmed_labels"] == {"charger_target": 1}
    assert statistics["strict_validation_identities"] == {"charger_target": 1}


def test_review_defaults_to_target_candidate_priority(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    other = _candidate(root, "a", "other.jpg", "other.mp4")
    target = _candidate(root, "b", "target.jpg", "target.mp4")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            [
                json.dumps({"image_path": str(other), "ai_recommendation": "other", "similarity": 0.9, "margin": 0.4, "top3": [{"label": "other", "score": 0.9}]}),
                json.dumps({"image_path": str(target), "ai_recommendation": "charger_target", "similarity": 0.7, "margin": 0.1, "top3": [{"label": "charger_target", "score": 0.7}]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    state = create_app(
        [root],
        tmp_path / "annotations.jsonl",
        predictions_path=predictions,
        evaluation_path=tmp_path / "missing.json",
    ).test_client().get("/api/state").get_json()
    assert state["candidate"]["filename"] == "target.jpg"
    assert state["candidate"]["session_id"] == "b"
    assert state["candidate"]["quality_score"] == 0.81
    assert state["candidate"]["detection_valid"] is True
    assert state["candidate"]["classification_usable"] is False
