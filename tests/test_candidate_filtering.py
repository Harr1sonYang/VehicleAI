from vehicleai.tracking.verification import assess_candidate_track


def diagnostic(**overrides):
    values = {
        "detection_count": 20,
        "continuity": 0.9,
        "mean_confidence": 0.7,
        "mean_area_ratio": 0.05,
        "center_x_std": 0.05,
    }
    values.update(overrides)
    return values


CONFIG = {
    "enabled": True,
    "min_track_detections": 5,
    "min_continuity": 0.35,
    "min_mean_confidence": 0.4,
    "interior_min_area_ratio": 0.25,
    "interior_max_horizontal_motion": 0.015,
    "secondary_verification": {"positive_threshold": 0.25, "negative_threshold": 0.05},
}


def verification(probability: float | None, available: bool = True):
    return {"available": available, "vehicle_probability": probability, "reason": None}


def test_sparse_seatbelt_like_detection_is_filtered_with_secondary_evidence() -> None:
    status, reasons = assess_candidate_track(
        diagnostic(
            detection_count=8,
            continuity=0.24,
            mean_confidence=0.45,
            mean_area_ratio=0.30,
            center_x_std=0.004,
        ),
        verification(0.01),
        CONFIG,
    )
    assert status == "filtered"
    assert "weak_temporal_support" in reasons
    assert "secondary_nonvehicle_evidence" in reasons


def test_large_stationary_interior_detection_is_filtered() -> None:
    status, reasons = assess_candidate_track(
        diagnostic(mean_area_ratio=0.50, center_x_std=0.001), verification(0.0), CONFIG
    )
    assert status == "filtered"
    assert "interior_like_geometry" in reasons


def test_low_confidence_distant_vehicle_is_not_automatically_filtered() -> None:
    status, _ = assess_candidate_track(
        diagnostic(mean_confidence=0.25, mean_area_ratio=0.002), verification(0.01), CONFIG
    )
    assert status == "suspected_false_positive"


def test_short_transient_vehicle_is_kept_when_secondary_evidence_is_positive() -> None:
    status, reasons = assess_candidate_track(
        diagnostic(detection_count=2, continuity=1.0, mean_confidence=0.8), verification(0.7), CONFIG
    )
    assert status == "accepted"
    assert "secondary_vehicle_evidence" in reasons


def test_short_transient_vehicle_with_negative_classifier_is_only_suspected() -> None:
    status, _ = assess_candidate_track(
        diagnostic(detection_count=3, continuity=1.0, mean_confidence=0.5, mean_area_ratio=0.07),
        verification(0.01),
        CONFIG,
    )
    assert status == "suspected_false_positive"


def test_verifier_unavailable_falls_back_without_hard_filtering() -> None:
    status, reasons = assess_candidate_track(
        diagnostic(detection_count=1, continuity=0.2), verification(None, available=False), CONFIG
    )
    assert status == "suspected_false_positive"
    assert "secondary_verification_unavailable" in reasons


def test_false_positive_filter_can_be_disabled() -> None:
    config = {**CONFIG, "enabled": False}
    status, reasons = assess_candidate_track(
        diagnostic(detection_count=1, mean_area_ratio=0.8, center_x_std=0.0),
        verification(0.0),
        config,
    )
    assert status == "accepted"
    assert reasons == ["filter_disabled"]
