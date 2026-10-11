"""Leakage-resistant source-grouped classification dataset preparation."""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

TRAINING_CLASSES = ("charger_target", "explorer_target", "durango_target", "other")
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class PlannedSample:
    image_path: Path
    assigned_class: str
    group: str
    split: str


def group_key(record: dict[str, Any]) -> str:
    """Prefer source/session metadata; fall back to the individual image."""
    source = record.get("source")
    if source:
        normalised_source = str(source).replace("\\", "/").casefold()
        return f"source:{normalised_source}"
    return f"image:{Path(record['image_path']).resolve().as_posix().casefold()}"


def assign_group(group: str, ratios: tuple[float, float, float], seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "val"
    return "test"


def build_split_plan(
    records: Iterable[dict[str, Any]],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> list[PlannedSample]:
    if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must be non-negative and sum to 1")
    plan: list[PlannedSample] = []
    for record in records:
        assigned_class = record.get("assigned_class")
        if assigned_class not in TRAINING_CLASSES:
            continue
        image_path = Path(record["image_path"])
        group = group_key(record)
        plan.append(PlannedSample(image_path, assigned_class, group, assign_group(group, ratios, seed)))
    return plan


def plan_summary(plan: Iterable[PlannedSample]) -> dict[str, dict[str, int]]:
    counts = Counter((sample.split, sample.assigned_class) for sample in plan)
    return {
        split: {label: counts[(split, label)] for label in TRAINING_CLASSES}
        for split in SPLITS
    }


def materialise_plan(plan: Iterable[PlannedSample], output_root: Path) -> int:
    samples = list(plan)
    existing_images = [
        path for path in output_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ] if output_root.exists() else []
    if existing_images:
        raise FileExistsError(
            f"Output dataset already contains images: {output_root}. "
            "Use a new or empty output directory to prevent stale split leakage."
        )
    missing = [sample.image_path for sample in samples if not sample.image_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Annotated image not found: {missing[0]}")
    for split in SPLITS:
        for assigned_class in TRAINING_CLASSES:
            (output_root / split / assigned_class).mkdir(parents=True, exist_ok=True)
    count = 0
    for sample in samples:
        destination_dir = output_root / sample.split / sample.assigned_class
        digest = hashlib.sha1(str(sample.image_path.resolve()).encode("utf-8")).hexdigest()[:10]
        destination = destination_dir / f"{sample.image_path.stem}_{digest}{sample.image_path.suffix.lower()}"
        shutil.copy2(sample.image_path, destination)
        count += 1
    return count
