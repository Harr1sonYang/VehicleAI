"""Durable orchestration helpers for dashcam video collection."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from vehicleai.classification.curation import AnnotationStore, discover_candidates
from vehicleai.classification.data_audit import (
    difference_hash,
    file_sha256,
    hamming_distance,
)
from vehicleai.classification.identity import VehicleIdentityStore, infer_identity

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ACTIVE_STATUSES = {"processing", "tracking", "deduplicating", "prelabeling"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def video_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session_id_for(path: Path, digest: str) -> str:
    """Use one conservative recording-session identity per source video."""
    safe_stem = "".join(character if character.isalnum() or character in "-_" else "-" for character in path.stem)
    safe_stem = safe_stem.strip("-")[:48] or "video"
    return f"{safe_stem}-{digest[:12]}"


class VideoQueue:
    """SQLite-backed queue that survives interruption without touching source videos."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialise(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    modified_ns INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    output_dir TEXT,
                    last_error TEXT,
                    discovered_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS video_sources (
                    source_path TEXT PRIMARY KEY,
                    file_size INTEGER NOT NULL,
                    modified_ns INTEGER NOT NULL,
                    video_id TEXT NOT NULL REFERENCES videos(video_id)
                );
                """
            )

    def discover(self, incoming_root: Path) -> dict[str, int]:
        root = Path(incoming_root).resolve()
        if not root.is_dir():
            root.mkdir(parents=True, exist_ok=True)
        found = added = duplicate_content = 0
        paths = sorted(
            (path.resolve() for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES),
            key=lambda item: item.as_posix().casefold(),
        )
        with self.connect() as connection:
            for path in paths:
                found += 1
                stat = path.stat()
                source_key = path.as_posix().casefold()
                cached = connection.execute(
                    "SELECT video_id FROM video_sources WHERE source_path=? AND file_size=? AND modified_ns=?",
                    (source_key, stat.st_size, stat.st_mtime_ns),
                ).fetchone()
                if cached:
                    continue
                digest = video_sha256(path)
                existing = connection.execute(
                    "SELECT video_id FROM videos WHERE video_id=?", (digest,)
                ).fetchone()
                now = utc_now()
                if existing:
                    duplicate_content += 1
                else:
                    connection.execute(
                        """INSERT INTO videos
                        (video_id, source_path, session_id, file_size, modified_ns, status, stage,
                         discovered_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, 'pending', 'discovered', ?, ?)""",
                        (
                            digest,
                            str(path),
                            session_id_for(path, digest),
                            stat.st_size,
                            stat.st_mtime_ns,
                            now,
                            now,
                        ),
                    )
                    added += 1
                connection.execute(
                    """INSERT INTO video_sources (source_path, file_size, modified_ns, video_id)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(source_path) DO UPDATE SET
                    file_size=excluded.file_size, modified_ns=excluded.modified_ns,
                    video_id=excluded.video_id""",
                    (source_key, stat.st_size, stat.st_mtime_ns, digest),
                )
        return {"found": found, "added": added, "duplicate_content": duplicate_content}

    def recover_interrupted(self) -> int:
        statuses = tuple(sorted(ACTIVE_STATUSES))
        placeholders = ",".join("?" for _ in statuses)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT video_id FROM videos WHERE status IN ({placeholders})", statuses
            ).fetchall()
            if rows:
                connection.execute(
                    f"""UPDATE videos SET status='pending', stage='recovered',
                    last_error='Previous run was interrupted; safe retry scheduled', updated_at=?
                    WHERE status IN ({placeholders})""",
                    (utc_now(), *statuses),
                )
        return len(rows)

    def next(self, *, retry_failed: bool = True, max_attempts: int = 3) -> dict[str, Any] | None:
        statuses = ("pending", "failed") if retry_failed else ("pending",)
        placeholders = ",".join("?" for _ in statuses)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"""SELECT * FROM videos WHERE status IN ({placeholders}) AND attempts < ?
                ORDER BY discovered_at, source_path LIMIT 1""",
                (*statuses, max_attempts),
            ).fetchone()
            if row is None:
                return None
            attempts = int(row["attempts"]) + 1
            connection.execute(
                """UPDATE videos SET status='processing', stage='claimed', attempts=?,
                last_error=NULL, updated_at=? WHERE video_id=?""",
                (attempts, utc_now(), row["video_id"]),
            )
            result = dict(row)
            result["attempts"] = attempts
            result["status"] = "processing"
            result["stage"] = "claimed"
            return result

    def update(
        self,
        video_id: str,
        *,
        status: str,
        stage: str,
        output_dir: Path | None = None,
        error: str | None = None,
    ) -> None:
        completed_at = utc_now() if status == "completed" else None
        with self.connect() as connection:
            connection.execute(
                """UPDATE videos SET status=?, stage=?, output_dir=COALESCE(?, output_dir),
                last_error=?, updated_at=?, completed_at=COALESCE(?, completed_at)
                WHERE video_id=?""",
                (
                    status,
                    stage,
                    str(output_dir.resolve()) if output_dir else None,
                    error,
                    utc_now(),
                    completed_at,
                    video_id,
                ),
            )

    def rows(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM videos ORDER BY discovered_at, source_path")]

    def summary(self) -> dict[str, Any]:
        rows = self.rows()
        statuses = Counter(str(row["status"]) for row in rows)
        return {
            "video_count": len(rows),
            "status_counts": dict(sorted(statuses.items())),
            "attempt_count": sum(int(row["attempts"]) for row in rows),
            "videos": rows,
        }


def _metadata_by_image(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for metadata_path in root.rglob("metadata.jsonl") if root.is_dir() else ():
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                crop = record.get("crop_path") or record.get("image_path")
                if crop:
                    result[Path(str(crop)).resolve().as_posix().casefold()] = record
    return result


def find_cross_video_duplicates(
    candidate_root: Path,
    *,
    near_duplicate_distance: int = 2,
) -> list[dict[str, Any]]:
    """Flag content duplicates across sources without changing vehicle identities."""
    root = Path(candidate_root).resolve()
    metadata = _metadata_by_image(root)
    candidates = []
    if root.is_dir():
        for candidate in discover_candidates(root):
            record = metadata.get(candidate.key, {})
            candidates.append((candidate, record))
    exact_index: dict[str, tuple[Path, str | None, str | None]] = {}
    previous: list[tuple[str, Path, str | None, str | None]] = []
    near_buckets: dict[tuple[int, int], list[int]] = {}
    records: list[dict[str, Any]] = []
    for candidate, metadata_record in candidates:
        source = candidate.source_video or candidate.source
        session_id = metadata_record.get("session_id")
        digest = file_sha256(candidate.image_path)
        dhash = difference_hash(candidate.image_path)
        duplicate_type = None
        duplicate_of = None
        other_source = None
        exact = exact_index.get(digest)
        if exact and exact[1] != source:
            duplicate_type, duplicate_of, other_source = "exact", str(exact[0]), exact[1]
        if duplicate_type is None:
            hash_value = int(dhash, 16)
            candidate_indices: set[int] = set()
            for chunk_index in range(4):
                chunk_value = (hash_value >> (chunk_index * 16)) & 0xFFFF
                candidate_indices.update(near_buckets.get((chunk_index, chunk_value), ()))
            for index in sorted(candidate_indices):
                old_hash, old_path, old_source, _old_session = previous[index]
                if old_source != source and hamming_distance(dhash, old_hash) <= near_duplicate_distance:
                    duplicate_type, duplicate_of, other_source = "near", str(old_path), old_source
                    break
        records.append(
            {
                "image_path": str(candidate.image_path),
                "source_video": source,
                "session_id": session_id,
                "sha256": digest,
                "dhash": dhash,
                "duplicate_status": duplicate_type or "unique",
                "duplicate_of": duplicate_of,
                "duplicate_source_video": other_source,
                "identity_action": "none",
                "note": "Content duplicate flag only; never a vehicle identity assertion.",
            }
        )
        exact_index.setdefault(digest, (candidate.image_path, source, session_id))
        previous.append((dhash, candidate.image_path, source, session_id))
        previous_index = len(previous) - 1
        hash_value = int(dhash, 16)
        for chunk_index in range(4):
            chunk_value = (hash_value >> (chunk_index * 16)) & 0xFFFF
            near_buckets.setdefault((chunk_index, chunk_value), []).append(previous_index)
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    materialised = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in materialised:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return len(materialised)


def collection_statistics(
    queue: VideoQueue,
    candidate_root: Path,
    annotations_path: Path,
    identities_path: Path,
) -> dict[str, Any]:
    queue_summary = queue.summary()
    track_count = candidate_count = 0
    for row in queue_summary["videos"]:
        if row["status"] != "completed" or not row.get("output_dir"):
            continue
        summary_path = Path(row["output_dir"]) / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            track_count += int(summary.get("track_count", 0))
            candidate_count += int(summary.get("selected_images", 0))

    annotations = AnnotationStore(annotations_path).latest
    identities = VehicleIdentityStore(identities_path)
    labels: Counter[str] = Counter()
    confirmed_identities: dict[str, set[str]] = {}
    root = Path(candidate_root).resolve()
    if root.is_dir():
        for candidate in discover_candidates(root):
            annotation = annotations.get(candidate.key)
            if not annotation:
                continue
            label = str(annotation.get("label") or annotation.get("assigned_class"))
            labels[label] += 1
            inferred = infer_identity(
                image_path=candidate.image_path,
                source_video=candidate.source_video,
                track_id=candidate.track_id,
                vehicle_id=candidate.vehicle_id,
                provider=candidate.provider,
                source_id=candidate.source_id,
            )
            identity = identities.resolve(inferred, candidate.image_path)
            if identity.status == "confirmed":
                confirmed_identities.setdefault(label, set()).add(identity.vehicle_identity_id)
    return {
        "generated_at": utc_now(),
        "queue": {key: value for key, value in queue_summary.items() if key != "videos"},
        "video_progress": [
            {
                "source_path": row["source_path"],
                "session_id": row["session_id"],
                "status": row["status"],
                "stage": row["stage"],
                "attempts": row["attempts"],
                "last_error": row["last_error"],
            }
            for row in queue_summary["videos"]
        ],
        "independent_tracks": track_count,
        "candidate_images": candidate_count,
        "human_confirmed_labels": dict(sorted(labels.items())),
        "strict_validation_identities": {
            label: len(values) for label, values in sorted(confirmed_identities.items())
        },
    }
