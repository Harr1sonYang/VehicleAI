"""Extract detected vehicle crops and JSONL metadata from images or video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from vehicleai.detection.detector import VehicleDetector, crop_detection, open_video_source
from vehicleai.utils.io import append_jsonl, ensure_directory

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Image, image directory, video, or camera index")
    parser.add_argument("--output-dir", type=Path, default=Path("data/candidates"))
    parser.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 1, ...")
    parser.add_argument("--frame-stride", type=int, default=5, help="For video, process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=0, help="For video, stop after N frames")
    return parser.parse_args()


def save_frame_crops(frame, detector, output_dir: Path, source: str, frame_index: int) -> int:
    detections, _ = detector.predict(frame)
    records = []
    for detection_index, detection in enumerate(detections):
        filename = f"{Path(source).stem}_f{frame_index:08d}_d{detection_index:03d}_{detection.class_name}.jpg"
        crop_path = output_dir / filename
        crop = crop_detection(frame, detection, float(detector.config.get("crop_padding", 0.0)))
        if not cv2.imwrite(str(crop_path), crop):
            raise RuntimeError(f"Unable to save crop: {crop_path}")
        records.append(
            {
                "source": source,
                "frame_index": frame_index,
                "crop_path": str(crop_path),
                **detection.to_dict(),
            }
        )
    append_jsonl(output_dir / "metadata.jsonl", records)
    return len(records)


def image_paths(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    if source.is_dir():
        return sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    return []


def main() -> int:
    args = parse_args()
    if args.frame_stride < 1:
        print("ERROR: --frame-stride must be at least 1", file=sys.stderr)
        return 2
    detector = VehicleDetector(args.config, args.device)
    output_dir = ensure_directory(args.output_dir)
    images = image_paths(Path(args.source)) if not args.source.isdigit() else []
    crop_count = 0

    if images:
        for frame_index, path in enumerate(images):
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"WARNING: Skipping unreadable image: {path}", file=sys.stderr)
                continue
            crop_count += save_frame_crops(frame, detector, output_dir, str(path), frame_index)
    else:
        capture = open_video_source(args.source)
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok or (args.max_frames and frame_index >= args.max_frames):
                    break
                if frame_index % args.frame_stride == 0:
                    crop_count += save_frame_crops(frame, detector, output_dir, args.source, frame_index)
                frame_index += 1
        finally:
            capture.release()

    print(f"Saved {crop_count} vehicle crop(s) to {output_dir}")
    print(f"Metadata: {output_dir / 'metadata.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

