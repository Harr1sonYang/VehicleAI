"""Scan a dashcam inbox and run the complete VehicleAI candidate pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from vehicleai.collection.road import (
    VideoQueue,
    collection_statistics,
    find_cross_video_duplicates,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=Path("configs/road_collection.yaml"))
    known, _ = preliminary.parse_known_args()
    config = {}
    if known.config.is_file():
        config = yaml.safe_load(known.config.read_text(encoding="utf-8")) or {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=known.config)
    parser.add_argument("--inbox", type=Path, default=Path(config.get("inbox", "data/incoming/videos")))
    parser.add_argument("--output-root", type=Path, default=Path(config.get("candidate_root", "data/candidates/road")))
    parser.add_argument("--state", type=Path, default=Path(config.get("queue_database", "data/road_collection/queue.sqlite3")))
    parser.add_argument(
        "--duplicates", type=Path, default=Path(config.get("duplicate_index", "data/road_collection/candidate_duplicates.jsonl"))
    )
    parser.add_argument("--statistics", type=Path, default=Path(config.get("statistics", "data/road_collection/statistics.json")))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/annotations.jsonl"))
    parser.add_argument(
        "--identities", type=Path, default=Path("data/annotations/vehicle_identities.jsonl")
    )
    parser.add_argument("--predictions", type=Path, default=Path("data/predictions/prelabels.jsonl"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame-stride", type=int, default=int(config.get("frame_stride", 1)))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=0, help="0 processes the entire queue")
    parser.add_argument("--max-attempts", type=int, default=int(config.get("max_attempts", 3)))
    parser.add_argument("--retry-failed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=bool(config.get("prelabel_offline", True)),
    )
    parser.add_argument("--skip-prelabel", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def _archive_partial(output_dir: Path, failed_root: Path, attempt: int) -> Path | None:
    if not output_dir.exists():
        return None
    failed_root.mkdir(parents=True, exist_ok=True)
    target = failed_root / f"{output_dir.name}-attempt-{attempt:02d}"
    counter = 1
    while target.exists():
        target = failed_root / f"{output_dir.name}-attempt-{attempt:02d}-{counter:02d}"
        counter += 1
    shutil.move(str(output_dir), str(target))
    return target


def _run_tracking(args: argparse.Namespace, row: dict, project_root: Path) -> Path:
    output_dir = args.output_root / row["session_id"]
    if (output_dir / "summary.json").is_file() and (output_dir / "metadata.jsonl").is_file():
        return output_dir
    if output_dir.exists():
        _archive_partial(output_dir, args.state.parent / "failed", int(row["attempts"]))
    command = [
        sys.executable,
        str(project_root / "scripts" / "track_video.py"),
        str(row["source_path"]),
        "--output-root",
        str(args.output_root),
        "--run-name",
        str(row["session_id"]),
        "--session-id",
        str(row["session_id"]),
        "--source-video-id",
        str(row["video_id"]),
        "--device",
        str(args.device),
        "--frame-stride",
        str(args.frame_stride),
        "--max-frames",
        str(args.max_frames),
    ]
    completed = subprocess.run(command, cwd=project_root, check=False)
    if completed.returncode:
        raise RuntimeError(f"track_video.py exited with code {completed.returncode}")
    if not (output_dir / "summary.json").is_file():
        raise RuntimeError("Tracking ended without a summary.json completion marker")
    return output_dir


def _run_prelabel(args: argparse.Namespace, project_root: Path) -> None:
    command = [
        sys.executable,
        str(project_root / "scripts" / "prelabel_candidates.py"),
        "--device",
        str(args.device),
        "--output",
        str(args.predictions),
        "--root",
        str(args.output_root),
        "--merge-existing",
    ]
    if args.offline:
        command.append("--offline")
    completed = subprocess.run(command, cwd=project_root, check=False)
    if completed.returncode:
        raise RuntimeError(f"prelabel_candidates.py exited with code {completed.returncode}")


def _write_statistics(args: argparse.Namespace, queue: VideoQueue) -> dict:
    statistics = collection_statistics(queue, args.output_root, args.annotations, args.identities)
    args.statistics.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.statistics.with_suffix(args.statistics.suffix + ".tmp")
    temporary.write_text(json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.statistics)
    return statistics


def main() -> int:
    args = parse_args()
    if args.frame_stride < 1 or args.max_frames < 0 or args.max_videos < 0 or args.max_attempts < 1:
        raise SystemExit("frame/max values are invalid")
    project_root = Path(__file__).resolve().parents[1]
    queue = VideoQueue(args.state)
    discovery = queue.discover(args.inbox)
    print(f"Scan: {discovery['found']} video(s), {discovery['added']} new, {discovery['duplicate_content']} duplicate copy/copies")
    if args.status_only:
        statistics = _write_statistics(args, queue)
        print(json.dumps(statistics, ensure_ascii=False, indent=2))
        return 0
    recovered = queue.recover_interrupted()
    if recovered:
        print(f"Recovered {recovered} interrupted queue item(s)")

    processed = failures = 0
    while not args.max_videos or processed < args.max_videos:
        row = queue.next(retry_failed=args.retry_failed, max_attempts=args.max_attempts)
        if row is None:
            break
        processed += 1
        print(f"[{processed}/{args.max_videos or 'all'}] Tracking {row['source_path']} (attempt {row['attempts']})")
        queue.update(row["video_id"], status="tracking", stage="yolo_bytetrack")
        output_dir = args.output_root / row["session_id"]
        try:
            output_dir = _run_tracking(args, row, project_root)
        except KeyboardInterrupt:
            archived = _archive_partial(output_dir, args.state.parent / "failed", int(row["attempts"]))
            queue.update(
                row["video_id"],
                status="pending",
                stage="interrupted",
                error=f"Interrupted; partial output preserved at {archived}" if archived else "Interrupted",
            )
            raise
        except Exception as error:
            archived = _archive_partial(output_dir, args.state.parent / "failed", int(row["attempts"]))
            detail = f"{error}; partial output preserved at {archived}" if archived else str(error)
            queue.update(row["video_id"], status="failed", stage="tracking_failed", error=detail)
            failures += 1
            print(f"FAILED: {row['source_path']}: {detail}", file=sys.stderr)
            continue
        queue.update(
            row["video_id"], status="completed", stage="tracking_complete", output_dir=output_dir
        )

    print("Checking exact and near duplicates across source videos...")
    duplicate_records = find_cross_video_duplicates(args.output_root)
    write_jsonl(duplicate_records, args.duplicates)
    if not args.skip_prelabel and queue.summary()["status_counts"].get("completed", 0):
        print("Running cached DINOv2 pre-labeling for all unreviewed candidates...")
        try:
            _run_prelabel(args, project_root)
        except Exception as error:
            print(f"PRELABEL FAILED: {error}", file=sys.stderr)
            failures += 1
    statistics = _write_statistics(args, queue)
    exact = sum(record["duplicate_status"] == "exact" for record in duplicate_records)
    near = sum(record["duplicate_status"] == "near" for record in duplicate_records)
    print(
        f"Complete: videos={statistics['queue']['video_count']}, tracks={statistics['independent_tracks']}, "
        f"candidates={statistics['candidate_images']}, cross-video exact={exact}, near={near}"
    )
    print(f"Review: {sys.executable} scripts/review_dataset.py --root {args.output_root} --predictions {args.predictions}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
