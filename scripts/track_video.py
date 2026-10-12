"""Track vehicles with ByteTrack and save representative review candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml

from vehicleai.detection.detector import open_video_source
from vehicleai.temporal.frame_selection import FrameSelector
from vehicleai.tracking.association import associate_track_fragments
from vehicleai.tracking.diagnostics import build_track_diagnostics, diagnostics_summary
from vehicleai.tracking.tracker import VehicleTracker
from vehicleai.tracking.verification import LightweightVehicleVerifier, assess_candidate_track
from vehicleai.utils.io import ensure_directory, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Input video file")
    parser.add_argument("--output-root", type=Path, default=Path("data/candidates/tracked"))
    parser.add_argument("--run-name", help="Unique output folder name; default includes UTC time")
    parser.add_argument("--detection-config", type=Path, default=Path("configs/detection.yaml"))
    parser.add_argument("--tracking-config", type=Path, default=Path("configs/tracking.yaml"))
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 1, ...")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--session-id", help="Stable recording-session identifier")
    parser.add_argument("--source-video-id", help="Stable content hash used by the batch queue")
    parser.add_argument("--save-annotated-video", action="store_true")
    parser.add_argument("--show", action="store_true", help="Show live tracking; press q to stop")
    return parser.parse_args()


def load_tracking_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Tracking config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if not args.source.is_file():
        print(f"ERROR: Video not found: {args.source}", file=sys.stderr)
        return 2
    if args.frame_stride < 1:
        print("ERROR: --frame-stride must be at least 1", file=sys.stderr)
        return 2

    config = load_tracking_config(args.tracking_config)
    run_name = args.run_name or f"{args.source.stem}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output_dir = args.output_root / run_name
    if output_dir.exists():
        print(f"ERROR: Output already exists; refusing to overwrite: {output_dir}", file=sys.stderr)
        return 2
    ensure_directory(output_dir)

    tracker = VehicleTracker(
        args.detection_config,
        args.device,
        str(config.get("tracker", "bytetrack.yaml")),
        tracking_confidence=float(config.get("tracking_confidence", 0.10)),
        agnostic_nms=bool(config.get("agnostic_nms", True)),
    )
    selector = FrameSelector(
        max_per_track=int(config.get("max_selected_per_track", 5)),
        min_width=int(config.get("min_crop_width", 64)),
        min_height=int(config.get("min_crop_height", 48)),
        min_area_ratio=float(config.get("min_area_ratio", 0.001)),
        min_sharpness=float(config.get("min_sharpness", 35.0)),
        min_difference=float(config.get("min_frame_difference", 0.06)),
        thumbnail_size=int(config.get("thumbnail_size", 64)),
    )
    capture = open_video_source(str(args.source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    annotated_path = output_dir / "tracked.mp4"
    if args.save_annotated_video:
        writer = cv2.VideoWriter(
            str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to create output video: {annotated_path}")

    frame_index = processed_frames = sampled_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or (args.max_frames and frame_index >= args.max_frames):
                break
            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            detections, annotated = tracker.track(frame)
            if frame_index % args.frame_stride == 0:
                for detection in detections:
                    selector.consider(
                        frame,
                        detection,
                        source=str(args.source),
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        padding=float(tracker.config.get("crop_padding", 0.0)),
                        session_id=args.session_id,
                        source_video_id=args.source_video_id,
                    )
                sampled_frames += 1
            if writer is not None:
                writer.write(annotated)
            if args.show:
                cv2.imshow("VehicleAI tracking - press q to stop", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            processed_frames += 1
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    diagnostic_config = config.get("diagnostics", {})
    diagnostics = build_track_diagnostics(
        selector.records,
        frame_width=width,
        frame_height=height,
        fps=fps,
        short_track_max_detections=int(diagnostic_config.get("short_track_max_detections", 4)),
    )
    association_config = config.get("association", {})
    associations = associate_track_fragments(
        selector.records,
        frame_width=width,
        frame_height=height,
        max_gap_frames=(
            int(association_config.get("max_gap_frames", 90))
            if bool(association_config.get("enabled", True))
            else -1
        ),
        max_spatial_distance=float(association_config.get("max_spatial_distance", 0.30)),
        min_scale_similarity=float(association_config.get("min_scale_similarity", 0.35)),
        suspected_score=float(association_config.get("suspected_score", 0.68)),
        auto_merge_score=float(association_config.get("auto_merge_score", 0.86)),
        min_auto_appearance=float(association_config.get("min_auto_appearance", 0.72)),
    )
    selected = selector.selected_records()
    selected_by_track: dict[int, list] = defaultdict(list)
    for record in selected:
        selected_by_track[record.detection.track_id].append(record)
    filter_config = config.get("false_positive_filter", {})
    secondary_config = filter_config.get("secondary_verification", {})
    verifier = LightweightVehicleVerifier(
        Path(str(secondary_config.get("model", "models/yolo11n-cls.pt"))),
        tracker.device,
        enabled=bool(secondary_config.get("enabled", True)),
    )
    quality_by_track: dict[int, dict] = {}
    for track_id, diagnostic in diagnostics.items():
        verification = verifier.verify(
            selected_by_track.get(track_id, []),
            max_samples=int(secondary_config.get("max_samples_per_track", 3)),
        )
        candidate_status, reasons = assess_candidate_track(diagnostic, verification, filter_config)
        quality_by_track[track_id] = {
            "candidate_status": candidate_status,
            "filter_reasons": reasons,
            "secondary_verification": verification,
        }
        diagnostic.update(quality_by_track[track_id])

    for record in selector.records:
        track_id = record.detection.track_id
        group_id = associations.group_by_track[track_id]
        quality = quality_by_track[track_id]
        record.vehicle_group_id = group_id
        record.group_track_ids = associations.group_members[group_id]
        record.association_status = associations.status_by_track[track_id]
        record.suspected_duplicate_track_ids = associations.suspected_by_track.get(track_id, [])
        record.candidate_status = quality["candidate_status"]
        record.candidate_filter_reasons = quality["filter_reasons"]
        record.track_statistics = {
            key: value
            for key, value in diagnostics[track_id].items()
            if key not in {"secondary_verification", "filter_reasons"}
        }
        verification = quality["secondary_verification"]
        record.secondary_verification = {
            "available": verification.get("available"),
            "reason": verification.get("reason"),
            "vehicle_probability": verification.get("vehicle_probability"),
        }

    for track_id, diagnostic in diagnostics.items():
        group_id = associations.group_by_track[track_id]
        diagnostic.update(
            {
                "vehicle_group_id": group_id,
                "group_track_ids": associations.group_members[group_id],
                "association_status": associations.status_by_track[track_id],
                "suspected_duplicate_track_ids": associations.suspected_by_track.get(track_id, []),
            }
        )

    for record in selected:
        track_dir = ensure_directory(output_dir / f"track_{record.detection.track_id:06d}")
        filename = f"f{record.frame_index:08d}_{record.detection.class_name}.jpg"
        crop_path = track_dir / filename
        if not cv2.imwrite(str(crop_path), record.crop):
            raise RuntimeError(f"Unable to save crop: {crop_path}")
        record.crop_path = str(crop_path)

    selected_metadata = [record.to_dict() for record in selected]
    selection_log = [record.to_dict() for record in selector.records]
    write_jsonl(output_dir / "metadata.jsonl", selected_metadata)
    write_jsonl(output_dir / "selection_log.jsonl", selection_log)
    write_jsonl(output_dir / "associations.jsonl", [item.to_dict() for item in associations.associations])
    write_json(
        output_dir / "track_diagnostics.json",
        {str(track_id): diagnostic for track_id, diagnostic in sorted(diagnostics.items())},
    )
    status_counts = Counter(record.candidate_status for record in selected)
    group_link_count = sum(max(0, len(members) - 1) for members in associations.group_members.values())
    summary = {
        "source": str(args.source),
        "session_id": args.session_id,
        "source_video_id": args.source_video_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": tracker.device,
        "processed_frames": processed_frames,
        "sampled_frames": sampled_frames,
        "frame_stride": args.frame_stride,
        **selector.summary(),
        **diagnostics_summary(diagnostics),
        "vehicle_group_count": len(associations.group_members),
        "inferred_id_switch_count": group_link_count,
        "suspected_duplicate_pair_count": sum(
            item.status == "suspected_duplicate" for item in associations.associations
        ),
        "candidate_status_counts": dict(sorted(status_counts.items())),
        "false_positive_filter_enabled": bool(filter_config.get("enabled", True)),
        "secondary_verifier_available": verifier.model is not None,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
