from pathlib import Path

import numpy as np

from vehicleai.classification.data_audit import AuditedSample
from vehicleai.classification.identity import (
    VehicleIdentityStore,
    find_cross_source_suggestions,
    infer_identity,
)


def test_same_track_is_confirmed_but_different_tracks_are_not_merged(tmp_path: Path) -> None:
    first = infer_identity(image_path=tmp_path / "a.jpg", source_video="drive.mp4", track_id=7)
    second = infer_identity(image_path=tmp_path / "b.jpg", source_video="drive.mp4", track_id=7)
    other_track = infer_identity(image_path=tmp_path / "c.jpg", source_video="drive.mp4", track_id=8)
    assert first.vehicle_identity_id == second.vehicle_identity_id
    assert first.status == "confirmed"
    assert first.evidence == "same_video_track"
    assert first.vehicle_identity_id != other_track.vehicle_identity_id


def test_unknown_identity_stays_unknown_without_evidence(tmp_path: Path) -> None:
    identity = infer_identity(image_path=tmp_path / "unknown.jpg")
    assert identity.status == "unknown_identity"
    assert identity.evidence == "insufficient_identity_evidence"


def test_manual_merge_split_and_suspicion_are_append_only(tmp_path: Path) -> None:
    path = tmp_path / "vehicle_identities.jsonl"
    left_path, right_path = tmp_path / "left.jpg", tmp_path / "right.jpg"
    left = infer_identity(image_path=left_path)
    right = infer_identity(image_path=right_path)
    store = VehicleIdentityStore(path)
    store.suggest_duplicate([left, right])
    assert store.resolve(left, left_path).status == "unknown_identity"
    merged = store.merge([left, right])
    assert merged["action"] == "confirm_merge"
    assert store.resolve(left, left_path).vehicle_identity_id == store.resolve(right, right_path).vehicle_identity_id
    store.split(left, left_path)
    assert store.resolve(left, left_path).vehicle_identity_id != store.resolve(right, right_path).vehicle_identity_id
    reloaded = VehicleIdentityStore(path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3
    assert reloaded.resolve(left, left_path).vehicle_identity_id != reloaded.resolve(right, right_path).vehicle_identity_id
    history = reloaded.history_for(left, left_path)
    assert [record["action"] for record in history] == ["suspected_duplicate", "confirm_merge", "split"]


def test_cross_source_similarity_only_creates_a_suggestion(tmp_path: Path) -> None:
    samples = [
        AuditedSample(tmp_path / "a.jpg", "other", "a" * 64, "a" * 16, "identity-a", True, "a.mp4", 1, identity_subject="track:a:1"),
        AuditedSample(tmp_path / "b.jpg", "other", "b" * 64, "b" * 16, "identity-b", True, "b.mp4", 2, identity_subject="track:b:2"),
    ]
    suggestions = find_cross_source_suggestions(samples, np.asarray([[1.0, 0.0], [0.99, 0.01]]), minimum_similarity=0.9)
    assert len(suggestions) == 1
    assert suggestions[0]["status"] == "suspected_duplicate"
    assert suggestions[0]["manually_confirmed"] is False
