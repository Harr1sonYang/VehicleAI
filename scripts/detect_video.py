"""Detect vehicles in a video file or USB camera stream."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from vehicleai.detection.detector import VehicleDetector, open_video_source
from vehicleai.utils.io import append_jsonl, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Video path or USB camera index, e.g. 0")
    parser.add_argument("--output-dir", type=Path, default=Path("results/videos"))
    parser.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 1, ...")
    parser.add_argument("--show", action="store_true", help="Show a live preview; press q to stop")
    parser.add_argument("--no-save", action="store_true", help="Do not save annotated video")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N source frames (0=no limit)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frame_stride < 1:
        print("ERROR: --frame-stride must be at least 1", file=sys.stderr)
        return 2
    capture = open_video_source(args.source)
    detector = VehicleDetector(args.config, args.device)
    output_dir = ensure_directory(args.output_dir)
    source_name = f"camera_{args.source}" if args.source.isdigit() else Path(args.source).stem
    metadata_path = output_dir / f"{source_name}_metadata.jsonl"

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    video_path = output_dir / f"{source_name}_detected.mp4"
    if not args.no_save:
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to create output video: {video_path}")

    frame_index = 0
    processed = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if args.max_frames and frame_index >= args.max_frames:
                break
            if frame_index % args.frame_stride:
                if writer is not None:
                    writer.write(frame)
                frame_index += 1
                continue
            detections, annotated = detector.predict(frame)
            timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            records = [
                {"frame_index": frame_index, "timestamp_ms": timestamp_ms, **item.to_dict()}
                for item in detections
            ]
            append_jsonl(metadata_path, records)
            if writer is not None:
                writer.write(annotated)
            if args.show:
                cv2.imshow("VehicleAI - press q to stop", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            processed += 1
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"Processed {processed} frame(s) using device {detector.device}.")
    if writer is not None:
        print(f"Annotated video: {video_path}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

