from pathlib import Path

import numpy as np

from vehicleai.classification.prelabel import BalancedKNNClassifier, merge_prediction_records


def test_balanced_class_scoring_is_not_dominated_by_other_count(tmp_path: Path) -> None:
    labels = ["charger_target"] * 2 + ["other"] * 20
    charger = np.asarray([[1.0, 0.0], [0.98, 0.02]], dtype=np.float32)
    others = np.asarray([[0.75, 0.66]] * 20, dtype=np.float32)
    vectors = np.vstack([charger, others])
    paths = [tmp_path / f"{index}.jpg" for index in range(len(labels))]
    classifier = BalancedKNNClassifier(k_per_class=2, similarity_threshold=0.5, margin_threshold=0.01)
    classifier.fit(vectors, labels, paths)
    result = classifier.predict_one(np.asarray([1.0, 0.0], dtype=np.float32))
    assert result.predicted_label == "charger_target"
    assert result.top3[0]["label"] == "charger_target"
    assert result.nearest_reference_path.endswith("0.jpg")


def test_classifier_abstains_on_low_margin() -> None:
    classifier = BalancedKNNClassifier(k_per_class=1, similarity_threshold=0.1, margin_threshold=0.05)
    classifier.fit(
        np.asarray([[1.0, 0.0], [0.999, 0.04]], dtype=np.float32),
        ["charger_target", "explorer_target"],
        ["charger.jpg", "explorer.jpg"],
    )
    result = classifier.predict_one(np.asarray([1.0, 0.02], dtype=np.float32))
    assert result.predicted_label == "unknown"
    assert len(result.top3) == 2


def test_incremental_predictions_replace_only_matching_image(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    existing = [
        {"image_path": str(first), "ai_recommendation": "other"},
        {"image_path": str(second), "ai_recommendation": "explorer_target"},
    ]
    updates = [{"image_path": str(first), "ai_recommendation": "charger_target"}]
    merged = merge_prediction_records(existing, updates)
    by_name = {Path(item["image_path"]).name: item for item in merged}
    assert by_name["first.jpg"]["ai_recommendation"] == "charger_target"
    assert by_name["second.jpg"]["ai_recommendation"] == "explorer_target"
