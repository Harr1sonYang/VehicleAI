from types import SimpleNamespace
from pathlib import Path

import torch
import yaml

from vehicleai.tracking.tracker import tracked_detections_from_result


def test_tracked_detections_from_result() -> None:
    boxes = SimpleNamespace(
        id=torch.tensor([4.0]),
        xyxy=torch.tensor([[10.2, 20.4, 90.7, 120.6]]),
        cls=torch.tensor([2.0]),
        conf=torch.tensor([0.875]),
    )
    result = SimpleNamespace(boxes=boxes, names={2: "car"})
    detections = tracked_detections_from_result(result)
    assert len(detections) == 1
    assert detections[0].track_id == 4
    assert detections[0].class_name == "car"
    assert detections[0].to_dict()["x2"] == 91


def test_unconfirmed_track_is_ignored() -> None:
    result = SimpleNamespace(boxes=SimpleNamespace(id=None), names={})
    assert tracked_detections_from_result(result) == []


def test_vehicleai_bytetrack_config_preserves_low_score_association() -> None:
    config = yaml.safe_load(Path("configs/bytetrack_vehicleai.yaml").read_text(encoding="utf-8"))
    tracking = yaml.safe_load(Path("configs/tracking.yaml").read_text(encoding="utf-8"))
    assert config["track_low_thresh"] < config["track_high_thresh"]
    assert tracking["tracking_confidence"] <= config["track_low_thresh"]
    assert config["new_track_thresh"] >= config["track_high_thresh"]
    assert config["track_buffer"] >= 60
    assert tracking["agnostic_nms"] is True
