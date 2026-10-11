import json
from pathlib import Path

from vehicleai.classification.curation import AnnotationStore, Candidate, discover_candidates


def test_annotation_store_resumes_and_latest_wins(tmp_path: Path) -> None:
    image = tmp_path / "crop.jpg"
    image.touch()
    store = AnnotationStore(tmp_path / "annotations.jsonl")
    candidate = Candidate(image, source="drive.mp4", frame_number=12, detection_confidence=0.91)
    store.record(candidate, "other")
    store.record(candidate, "unknown")

    resumed = AnnotationStore(store.path)
    assert resumed.is_annotated(candidate)
    assert resumed.latest[candidate.key]["assigned_class"] == "unknown"
    assert len(store.path.read_text(encoding="utf-8").splitlines()) == 2


def test_discover_candidates_uses_v01_metadata(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    image = candidate_dir / "crop.jpg"
    image.touch()
    metadata = {
        "source": "session-a.mp4",
        "frame_index": 25,
        "crop_path": str(image),
        "confidence": 0.8,
    }
    (candidate_dir / "metadata.jsonl").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    candidates = discover_candidates(candidate_dir)
    assert len(candidates) == 1
    assert candidates[0].source == "session-a.mp4"
    assert candidates[0].source_video == "session-a.mp4"
    assert candidates[0].frame_number == 25
    assert candidates[0].detection_confidence == 0.8


def test_discover_candidates_uses_web_collection_metadata(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "web" / "charger_target"
    candidate_dir.mkdir(parents=True)
    image = candidate_dir / "web_charger_a83f219c1234.jpg"
    image.touch()
    metadata = {
        "filename": image.name,
        "image_path": image.name,
        "class_hint": "charger_target",
        "label": None,
        "provider": "wikimedia_commons",
        "query": "Dodge Charger rear view",
        "source_url": "https://upload.wikimedia.org/example.jpg",
        "source_page_url": "https://commons.wikimedia.org/?curid=123",
        "sha256": "a83f219c1234567890",
        "phash": "0123456789abcdef",
    }
    (candidate_dir / "metadata.jsonl").write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    candidates = discover_candidates(tmp_path / "web")

    assert len(candidates) == 1
    assert candidates[0].class_hint == "charger_target"
    assert candidates[0].provider == "wikimedia_commons"
    assert candidates[0].source_page_url == metadata["source_page_url"]
    assert candidates[0].sha256 == metadata["sha256"]
