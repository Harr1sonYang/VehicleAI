"""Small adapter that adds persistent ByteTrack IDs to YOLO detections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vehicleai.detection.detector import VehicleDetector


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tracked_detections_from_result(result: Any) -> list[TrackedDetection]:
    """Convert one Ultralytics tracking result into stable plain records."""
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []
    track_ids = boxes.id.int().cpu().tolist()
    xyxy = boxes.xyxy.cpu().tolist()
    class_ids = boxes.cls.int().cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    detections: list[TrackedDetection] = []
    for track_id, coordinates, class_id, confidence in zip(
        track_ids, xyxy, class_ids, confidences, strict=True
    ):
        x1, y1, x2, y2 = (int(round(value)) for value in coordinates)
        detections.append(
            TrackedDetection(
                track_id=int(track_id),
                class_id=int(class_id),
                class_name=str(result.names[int(class_id)]),
                confidence=float(confidence),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return detections


class VehicleTracker:
    """Reuse VehicleDetector configuration/model and enable ByteTrack state."""

    def __init__(
        self,
        detection_config: Path,
        device: str = "auto",
        tracker_config: str = "bytetrack.yaml",
        tracking_confidence: float | None = None,
        agnostic_nms: bool = True,
    ) -> None:
        self.detector = VehicleDetector(detection_config, device)
        self.tracker_config = tracker_config
        self.tracking_confidence = tracking_confidence
        self.agnostic_nms = agnostic_nms

    @property
    def device(self) -> str:
        return self.detector.device

    @property
    def config(self) -> dict[str, Any]:
        return self.detector.config

    def track(self, frame: Any, verbose: bool = False) -> tuple[list[TrackedDetection], Any]:
        results = self.detector.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=float(self.tracking_confidence or self.config["confidence"]),
            iou=float(self.config["iou"]),
            imgsz=int(self.config["image_size"]),
            classes=self.detector.class_ids,
            agnostic_nms=self.agnostic_nms,
            device=self.device,
            verbose=verbose,
        )
        result = results[0]
        return tracked_detections_from_result(result), result.plot()
