import hashlib
import json
from pathlib import Path

import pytest

from vehicleai.classification.curation import AnnotationStore, Candidate
from vehicleai.classification.manifest import (
    build_manifest_from_annotations,
    manifest_summary,
    read_curated_manifest,
)


def test_manifest_uses_latest_label_and_references_original(tmp_path: Path) -> None:
    image = tmp_path / "candidates" / "candidate.jpg"
    image.parent.mkdir()
    original_bytes = b"unchanged-candidate-image"
    image.write_bytes(original_bytes)
    annotations = tmp_path / "annotations" / "annotations.jsonl"
    store = AnnotationStore(annotations)
    candidate = Candidate(
        image,
        source="session-a.mp4",
        frame_number=42,
        track_id=7,
        class_hint="charger_target",
        provider="local_video",
        source_id="session-a",
        vehicle_id="vehicle-17",
        source_video="session-a.mp4",
    )
    store.record(candidate, "unknown")
    latest = store.record(candidate, "charger_target")
    output = tmp_path / "curated" / "curated_manifest.jsonl"

    records = build_manifest_from_annotations(annotations, output)

    assert len(records) == 1
    assert records[0] == {
        "original_path": str(image.resolve()),
        "final_label": "charger_target",
        "class_hint": "charger_target",
        "source": "session-a.mp4",
        "provider": "local_video",
        "source_id": "session-a",
        "vehicle_id": "vehicle-17",
        "source_video": "session-a.mp4",
        "frame_number": 42,
        "track_id": 7,
        "sha256": hashlib.sha256(original_bytes).hexdigest(),
        "annotation_timestamp": latest["timestamp"],
    }
    assert image.read_bytes() == original_bytes
    assert len(annotations.read_text(encoding="utf-8").splitlines()) == 2
    assert not (output.parent / "charger_target").exists()


def test_manifest_summary_includes_all_logical_categories(tmp_path: Path) -> None:
    manifest = tmp_path / "curated_manifest.jsonl"
    manifest.write_text(
        json.dumps({"original_path": "a.jpg", "final_label": "other"}) + "\n"
        + json.dumps({"original_path": "b.jpg", "final_label": "unusable"}) + "\n",
        encoding="utf-8",
    )

    summary = manifest_summary(read_curated_manifest(manifest))

    assert summary == {
        "charger_target": 0,
        "explorer_target": 0,
        "durango_target": 0,
        "other": 1,
        "unknown": 0,
        "unusable": 1,
    }


def test_missing_original_does_not_replace_existing_manifest(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.jsonl"
    missing = tmp_path / "missing.jpg"
    annotations.write_text(
        json.dumps(
            {
                "image_path": str(missing),
                "label": "other",
                "timestamp": "2026-10-09T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "curated_manifest.jsonl"
    output.write_text("existing-manifest\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        build_manifest_from_annotations(annotations, output)

    assert output.read_text(encoding="utf-8") == "existing-manifest\n"
