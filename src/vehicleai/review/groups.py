"""Append-only manual vehicle-group merge/split decisions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TrackGroupStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._overrides: dict[tuple[str, int], str] = {}
        self._load()

    @staticmethod
    def _source_key(source: str | None) -> str:
        return str(source or "unknown").replace("\\", "/").casefold()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                source = self._source_key(record.get("source_video"))
                group_id = str(record["vehicle_group_id"])
                for track_id in record["track_ids"]:
                    self._overrides[(source, int(track_id))] = group_id

    def group_for(self, source_video: str | None, track_id: int | str | None, automatic: str | None) -> str | None:
        if track_id is None:
            return automatic
        return self._overrides.get((self._source_key(source_video), int(track_id)), automatic)

    def _append(self, action: str, source_video: str, track_ids: list[int], group_id: str) -> dict[str, Any]:
        record = {
            "action": action,
            "source_video": source_video,
            "track_ids": sorted(set(track_ids)),
            "vehicle_group_id": group_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        source_key = self._source_key(source_video)
        for track_id in record["track_ids"]:
            self._overrides[(source_key, track_id)] = group_id
        return record

    def merge(self, source_video: str, track_ids: list[int]) -> dict[str, Any]:
        unique = sorted(set(int(value) for value in track_ids))
        if len(unique) < 2:
            raise ValueError("At least two track IDs are required to merge")
        digest = hashlib.sha1(f"{self._source_key(source_video)}:{unique}".encode("utf-8")).hexdigest()[:12]
        return self._append("merge", source_video, unique, f"manual_group_{digest}")

    def split(self, source_video: str, track_id: int) -> dict[str, Any]:
        digest = hashlib.sha1(f"{self._source_key(source_video)}:{track_id}".encode("utf-8")).hexdigest()[:12]
        return self._append("split", source_video, [int(track_id)], f"manual_track_{digest}")
