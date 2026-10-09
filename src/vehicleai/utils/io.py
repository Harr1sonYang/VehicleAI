"""Output and metadata helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

