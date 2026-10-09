"""Thin wrapper around an Ultralytics YOLO detector."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
import yaml

# Keep third-party runtime settings inside the project. This avoids depending on
# a writable Windows roaming profile and is harmless on Linux/Jetson.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))

from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = {"model", "confidence", "iou", "image_size", "vehicle_class_ids"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")
    return config


def select_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


class VehicleDetector:
    def __init__(self, config_path: Path, device: str = "auto") -> None:
        self.config = load_config(config_path)
        self.device = select_device(device)
        self.model = YOLO(str(self.config["model"]))
        self.class_ids = [int(value) for value in self.config["vehicle_class_ids"]]

    def predict(self, frame: Any, verbose: bool = False) -> tuple[list[Detection], Any]:
        results = self.model.predict(
            source=frame,
            conf=float(self.config["confidence"]),
            iou=float(self.config["iou"]),
            imgsz=int(self.config["image_size"]),
            classes=self.class_ids,
            device=self.device,
            verbose=verbose,
        )
        result = results[0]
        detections: list[Detection] = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
                class_id = int(box.cls.item())
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=str(result.names[class_id]),
                        confidence=float(box.conf.item()),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )
        return detections, result.plot()


def crop_detection(frame: Any, detection: Detection, padding: float = 0.0) -> Any:
    height, width = frame.shape[:2]
    box_width = detection.x2 - detection.x1
    box_height = detection.y2 - detection.y1
    pad_x = int(box_width * padding)
    pad_y = int(box_height * padding)
    x1 = max(0, detection.x1 - pad_x)
    y1 = max(0, detection.y1 - pad_y)
    x2 = min(width, detection.x2 + pad_x)
    y2 = min(height, detection.y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Detection produced an empty crop")
    return frame[y1:y2, x1:x2]


def open_video_source(source: str) -> cv2.VideoCapture:
    parsed_source: int | str = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(parsed_source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")
    return capture
