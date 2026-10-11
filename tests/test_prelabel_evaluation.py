from pathlib import Path

import numpy as np

from vehicleai.classification.data_audit import AuditedSample
from vehicleai.classification.evaluation import evaluate_split, split_reference_validation


def sample(name: str, label: str, group: str, confirmed: bool = True) -> AuditedSample:
    return AuditedSample(Path(name), label, name * 8, name * 2, group, confirmed)


def test_validation_split_never_crosses_identity_groups_and_ignores_unconfirmed() -> None:
    samples = [
        sample("a", "other", "group-a"),
        sample("b", "other", "group-a"),
        sample("c", "other", "group-b"),
        sample("d", "charger_target", "unverified", False),
    ]
    reference, validation = split_reference_validation(samples, validation_fraction=0.5)
    reference_groups = {samples[index].identity_group for index in reference}
    validation_groups = {samples[index].identity_group for index in validation}
    assert not reference_groups & validation_groups
    assert 3 in reference
    assert 3 not in validation


def test_evaluation_reports_missing_class_support_as_unreliable() -> None:
    samples = [sample("a", "other", "a"), sample("b", "other", "b"), sample("c", "charger_target", "u", False)]
    vectors = np.asarray([[0.0, 1.0], [0.0, 0.9], [1.0, 0.0]], dtype=np.float32)
    report = evaluate_split(samples, vectors, k_per_class=1, similarity_threshold=0.1, margin_threshold=0.0)
    assert report["identity_overlap"] == []
    assert report["metrics"]["per_class"]["charger_target"]["support"] == 0
    assert report["metrics"]["per_class"]["charger_target"]["recall"] is None
