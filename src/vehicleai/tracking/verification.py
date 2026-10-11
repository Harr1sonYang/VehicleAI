"""Optional lightweight candidate verification and conservative quality gating."""

from __future__ import annotations

from pathlib import Path
from typing import Any


VEHICLE_LABEL_PARTS = (
    "ambulance",
    "beach_wagon",
    "cab",
    "convertible",
    "jeep",
    "limousine",
    "minibus",
    "minivan",
    "model_t",
    "moving_van",
    "pickup",
    "police_van",
    "racer",
    "recreational_vehicle",
    "school_bus",
    "sports_car",
    "taxicab",
    "tow_truck",
    "trailer_truck",
    "trolleybus",
)


class LightweightVehicleVerifier:
    """Use a tiny ImageNet classifier as supporting evidence, never sole truth."""

    def __init__(self, model_path: Path, device: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.device = device
        self.model = None
        self.error: str | None = None
        if enabled:
            try:
                from ultralytics import YOLO

                self.model = YOLO(str(model_path))
            except Exception as error:  # fallback is a documented feature
                self.error = str(error)

    def verify(self, records: list[Any], max_samples: int = 3) -> dict[str, Any]:
        if not self.enabled:
            return {"available": False, "reason": "disabled", "vehicle_probability": None, "predictions": []}
        if self.model is None:
            return {"available": False, "reason": self.error or "unavailable", "vehicle_probability": None, "predictions": []}
        samples = sorted(records, key=lambda item: item.quality_score, reverse=True)[:max_samples]
        if not samples:
            return {"available": True, "reason": "no_samples", "vehicle_probability": 0.0, "predictions": []}
        try:
            results = self.model.predict([item.crop for item in samples], device=self.device, verbose=False)
        except Exception as error:
            return {"available": False, "reason": str(error), "vehicle_probability": None, "predictions": []}
        predictions: list[dict[str, Any]] = []
        probabilities: list[float] = []
        for result in results:
            top5 = [
                {"label": str(result.names[index]), "probability": float(result.probs.data[index])}
                for index in result.probs.top5
            ]
            vehicle_probability = sum(
                item["probability"]
                for item in top5
                if any(part in item["label"].casefold() for part in VEHICLE_LABEL_PARTS)
            )
            probabilities.append(vehicle_probability)
            predictions.append({"vehicle_probability": vehicle_probability, "top5": top5})
        return {
            "available": True,
            "reason": None,
            "vehicle_probability": max(probabilities, default=0.0),
            "predictions": predictions,
        }


def assess_candidate_track(
    diagnostic: dict[str, Any],
    verification: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str, list[str]]:
    """Return accepted/suspected/filtered using combined evidence."""
    if not bool(config.get("enabled", True)):
        return "accepted", ["filter_disabled"]
    weak_temporal = (
        diagnostic["detection_count"] < int(config.get("min_track_detections", 5))
        or diagnostic["continuity"] < float(config.get("min_continuity", 0.35))
    )
    low_confidence = diagnostic["mean_confidence"] < float(config.get("min_mean_confidence", 0.40))
    interior_like = (
        diagnostic["mean_area_ratio"] >= float(config.get("interior_min_area_ratio", 0.25))
        and diagnostic["center_x_std"] <= float(config.get("interior_max_horizontal_motion", 0.015))
    )
    reasons = []
    if weak_temporal:
        reasons.append("weak_temporal_support")
    if low_confidence:
        reasons.append("low_mean_confidence")
    if interior_like:
        reasons.append("interior_like_geometry")

    probability = verification.get("vehicle_probability")
    secondary = config.get("secondary_verification", {})
    positive = verification.get("available") and probability is not None and probability >= float(
        secondary.get("positive_threshold", 0.25)
    )
    negative = verification.get("available") and probability is not None and probability <= float(
        secondary.get("negative_threshold", 0.05)
    )
    if positive:
        reasons.append("secondary_vehicle_evidence")
        return "accepted", reasons
    if negative:
        reasons.append("secondary_nonvehicle_evidence")
    elif not verification.get("available"):
        reasons.append("secondary_verification_unavailable")

    # Never reject on low confidence, short duration, geometry, or classifier alone.
    if negative and interior_like:
        return "filtered", reasons
    if weak_temporal or interior_like or low_confidence:
        return "suspected_false_positive", reasons
    return "accepted", reasons
