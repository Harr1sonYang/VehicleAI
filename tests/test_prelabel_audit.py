import json
from pathlib import Path

from PIL import Image

from vehicleai.classification.data_audit import audit_human_samples


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def test_audit_excludes_non_training_filtered_missing_and_conflicting_duplicates(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    paths = []
    for index in range(6):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (20, 20), (index * 30, 10, 200 - index * 20)).save(path)
        paths.append(path)
    paths[4].write_bytes(paths[3].read_bytes())
    labels = ["charger_target", "unknown", "other", "other", "charger_target", "durango_target"]
    statuses = [None, None, "filtered", None, None, None]
    for index, (path, label, status) in enumerate(zip(paths, labels, statuses)):
        record = {"image_path": str(path), "label": label, "assigned_class": label, "timestamp": "now"}
        if status:
            record["candidate_status"] = status
        append_jsonl(annotations, record)
        append_jsonl(
            manifest,
            {"original_path": str(path), "final_label": label, "annotation_timestamp": "now", "sha256": None},
        )
    missing = tmp_path / "missing.png"
    append_jsonl(annotations, {"image_path": str(missing), "label": "other", "timestamp": "now"})
    append_jsonl(manifest, {"original_path": str(missing), "final_label": "other", "annotation_timestamp": "now"})

    result = audit_human_samples(manifest, annotations, near_duplicate_distance=-1)
    assert [sample.label for sample in result.samples] == ["charger_target", "durango_target"]
    reasons = {record["reason"] for record in result.excluded}
    assert "excluded_label:unknown" in reasons
    assert "filtered_candidate" in reasons
    assert "missing_image" in reasons
    assert "duplicate_cluster_label_conflict" in reasons
    assert all(not sample.identity_confirmed for sample in result.samples)
