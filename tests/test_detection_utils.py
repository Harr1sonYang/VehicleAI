from pathlib import Path

import numpy as np

from vehicleai.detection.detector import Detection, crop_detection, load_config


def test_load_config() -> None:
    config = load_config(Path("configs/detection.yaml"))
    assert config["vehicle_class_ids"] == [2, 5, 7]


def test_crop_detection_with_padding_is_clamped() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    detection = Detection(2, "car", 0.9, 0, 0, 50, 50)
    crop = crop_detection(frame, detection, padding=0.2)
    assert crop.shape == (60, 60, 3)

