"""Detect road vehicles in one image and save annotation plus metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from vehicleai.detection.detector import VehicleDetector, crop_detection
from vehicleai.utils.io import ensure_directory, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument("--output-dir", type=Path, default=Path("results/images"))
    parser.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 1, ...")
    parser.add_argument("--save-crops", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"ERROR: Image not found: {args.input}", file=sys.stderr)
        return 2
    frame = cv2.imread(str(args.input))
    if frame is None:
        print(f"ERROR: Unable to decode image: {args.input}", file=sys.stderr)
        return 2

    detector = VehicleDetector(args.config, args.device)
    detections, annotated = detector.predict(frame)
    output_dir = ensure_directory(args.output_dir)
    annotated_path = output_dir / f"{args.input.stem}_detected.jpg"
    cv2.imwrite(str(annotated_path), annotated)

    records = []
    for index, detection in enumerate(detections):
        record = detection.to_dict()
        if args.save_crops:
            crop_dir = ensure_directory(output_dir / "crops")
            crop_path = crop_dir / f"{args.input.stem}_{index:04d}_{detection.class_name}.jpg"
            crop = crop_detection(frame, detection, float(detector.config.get("crop_padding", 0.0)))
            cv2.imwrite(str(crop_path), crop)
            record["crop_path"] = str(crop_path)
        records.append(record)

    metadata_path = output_dir / f"{args.input.stem}_metadata.json"
    write_json(metadata_path, {"source": str(args.input), "device": detector.device, "detections": records})
    print(f"Detected {len(detections)} vehicle(s).")
    print(f"Annotated image: {annotated_path}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

