import json
from pathlib import Path

from PIL import Image

from vehicleai.classification.blind_test import (
    evaluate_blind_records,
    load_trusted_labels,
    reject_reference_overlaps,
)
from vehicleai.classification.data_audit import file_sha256


def test_blind_images_cannot_reenter_the_reference_library(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    duplicate = tmp_path / "blind-copy.png"
    independent = tmp_path / "independent.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(reference)
    duplicate.write_bytes(reference.read_bytes())
    Image.new("RGB", (16, 16), (90, 80, 70)).save(independent)
    accepted, rejected = reject_reference_overlaps(
        [duplicate, independent], {file_sha256(reference)}
    )
    assert accepted == [independent]
    assert rejected[0]["prediction_status"] == "rejected_reference_overlap"


def test_blind_metrics_require_a_confirmed_identity_human_label(tmp_path: Path) -> None:
    image = tmp_path / "blind.jpg"
    predictions = [
        {
            "image_path": str(image),
            "ai_recommendation": "other",
            "top3": [{"label": "other"}, {"label": "charger_target"}],
        }
    ]
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        json.dumps({"image_path": str(image), "label": "other", "identity_confirmed": False}) + "\n",
        encoding="utf-8",
    )
    assert evaluate_blind_records(predictions, load_trusted_labels(labels_path))["trusted_evaluated_samples"] == 0
    labels_path.write_text(
        json.dumps({"image_path": str(image), "label": "other", "identity_confirmed": True}) + "\n",
        encoding="utf-8",
    )
    report = evaluate_blind_records(predictions, load_trusted_labels(labels_path))
    assert report["trusted_evaluated_samples"] == 1
    assert report["metrics"]["per_class"]["other"]["f1"] == 1.0
