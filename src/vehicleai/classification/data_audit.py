"""Build a conservative, leakage-aware view of human-reviewed images."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError

from .curation import AnnotationStore
from .identity import VehicleIdentityStore, infer_identity
from .manifest import read_curated_manifest

TRAINING_LABELS = ("charger_target", "explorer_target", "durango_target", "other")


def normalise_path(value: str | Path) -> str:
    return Path(value).resolve().as_posix().casefold()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path, hash_size: int = 8) -> str:
    """Return a small perceptual hash without requiring an extra dependency."""
    with Image.open(path) as image:
        grey = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(grey.get_flattened_data())
    value = 0
    width = hash_size + 1
    for row in range(hash_size):
        for column in range(hash_size):
            value = (value << 1) | int(pixels[row * width + column] > pixels[row * width + column + 1])
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


@dataclass(frozen=True)
class AuditedSample:
    image_path: Path
    label: str
    sha256: str
    dhash: str
    identity_group: str
    identity_confirmed: bool
    source_video: str | None = None
    track_id: int | str | None = None
    vehicle_group_id: str | None = None
    identity_status: str = "unknown_identity"
    identity_evidence: str = "insufficient_identity_evidence"
    identity_subject: str = ""

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["image_path"] = str(self.image_path)
        return record


@dataclass(frozen=True)
class AuditResult:
    samples: tuple[AuditedSample, ...]
    excluded: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def audit_human_samples(
    manifest_path: Path,
    annotations_path: Path,
    *,
    near_duplicate_distance: int = 2,
    identities_path: Path = Path("data/annotations/vehicle_identities.jsonl"),
) -> AuditResult:
    """Return only existing, manually reviewed four-class images.

    Prediction files are deliberately not accepted as input. Exact/near
    duplicates are assigned one identity group, and label-conflicting clusters
    are excluded rather than guessed.
    """
    manifest = read_curated_manifest(manifest_path)
    latest = AnnotationStore(annotations_path).latest
    identity_store = VehicleIdentityStore(identities_path)
    excluded: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []

    for record in manifest:
        path_value = record.get("original_path") or record.get("image_path")
        path = Path(str(path_value)).resolve() if path_value else Path()
        annotation = latest.get(normalise_path(path)) if path_value else None
        label = annotation.get("label") if annotation else None
        reason = None
        if annotation is None:
            reason = "missing_human_annotation"
        elif label not in TRAINING_LABELS:
            reason = f"excluded_label:{label}"
        elif annotation.get("candidate_status") == "filtered":
            reason = "filtered_candidate"
        elif not path.is_file():
            reason = "missing_image"
        elif record.get("final_label") != label:
            reason = "manifest_annotation_mismatch"
        if reason:
            excluded.append({"image_path": str(path), "reason": reason})
            continue
        try:
            sha256 = record.get("sha256") or annotation.get("sha256") or file_sha256(path)
            dhash = annotation.get("phash") or difference_hash(path)
        except (OSError, UnidentifiedImageError) as error:
            excluded.append({"image_path": str(path), "reason": f"unreadable_image:{type(error).__name__}"})
            continue
        source_video = annotation.get("source_video") or record.get("source_video")
        track_id = annotation.get("track_id", record.get("track_id"))
        inferred = infer_identity(
            image_path=path,
            source_video=source_video,
            track_id=track_id,
            vehicle_id=annotation.get("vehicle_id") or record.get("vehicle_id"),
            provider=annotation.get("provider") or record.get("provider"),
            source_id=annotation.get("source_id") or record.get("source_id"),
            source_identity_confirmed=bool(
                annotation.get("source_identity_confirmed") or record.get("source_identity_confirmed")
            ),
        )
        identity = identity_store.resolve(inferred, path)
        raw.append(
            {
                "path": path,
                "label": label,
                "sha256": str(sha256).lower(),
                "dhash": str(dhash).lower(),
                "identity": identity.vehicle_identity_id,
                "identity_status": identity.status,
                "identity_evidence": identity.evidence,
                "identity_subject": identity.subject_key,
                "confirmed": identity.status == "confirmed",
                "source_video": source_video,
                "track_id": track_id,
                "vehicle_group_id": annotation.get("vehicle_group_id") or record.get("vehicle_group_id"),
            }
        )

    labels_by_hash: dict[str, set[str]] = defaultdict(set)
    for item in raw:
        labels_by_hash[item["sha256"]].add(item["label"])
    conflicting_hashes = {digest for digest, labels in labels_by_hash.items() if len(labels) > 1}
    if conflicting_hashes:
        clean_raw: list[dict[str, Any]] = []
        for item in raw:
            if item["sha256"] in conflicting_hashes:
                excluded.append({"image_path": str(item["path"]), "reason": "duplicate_cluster_label_conflict"})
            else:
                clean_raw.append(item)
        raw = clean_raw

    dsu = _DisjointSet(len(raw))
    by_sha: dict[str, int] = {}
    by_identity: dict[str, int] = {}
    for index, item in enumerate(raw):
        if item["sha256"] in by_sha:
            dsu.union(index, by_sha[item["sha256"]])
        else:
            by_sha[item["sha256"]] = index
        if item["identity"] in by_identity:
            dsu.union(index, by_identity[item["identity"]])
        else:
            by_identity[item["identity"]] = index

    # The collection is small enough for a conservative all-pairs perceptual
    # duplicate check. Only same-label images are joined automatically.
    for left in range(len(raw)):
        for right in range(left + 1, len(raw)):
            if raw[left]["label"] == raw[right]["label"] and hamming_distance(raw[left]["dhash"], raw[right]["dhash"]) <= near_duplicate_distance:
                dsu.union(left, right)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(raw)):
        clusters[dsu.find(index)].append(index)

    samples: list[AuditedSample] = []
    duplicate_rows = 0
    conflict_rows = sum(record["reason"] == "duplicate_cluster_label_conflict" for record in excluded)
    for member_indices in clusters.values():
        labels = {raw[index]["label"] for index in member_indices}
        if len(labels) > 1:
            conflict_rows += len(member_indices)
            for index in member_indices:
                excluded.append({"image_path": str(raw[index]["path"]), "reason": "duplicate_cluster_label_conflict"})
            continue
        # Keep one physical copy per content hash, but preserve multiple views
        # from a confirmed vehicle group under the same group key.
        seen_hashes: set[str] = set()
        group_token = hashlib.sha256(
            "|".join(sorted(raw[index]["identity"] for index in member_indices)).encode("utf-8")
        ).hexdigest()[:16]
        identity_ids = {raw[index]["identity"] for index in member_indices}
        group_confirmed = all(raw[index]["confirmed"] for index in member_indices) and len(identity_ids) == 1
        for index in sorted(member_indices, key=lambda value: str(raw[value]["path"]).casefold()):
            item = raw[index]
            if item["sha256"] in seen_hashes:
                duplicate_rows += 1
                excluded.append({"image_path": str(item["path"]), "reason": "exact_duplicate"})
                continue
            seen_hashes.add(item["sha256"])
            samples.append(
                AuditedSample(
                    image_path=item["path"],
                    label=item["label"],
                    sha256=item["sha256"],
                    dhash=item["dhash"],
                    identity_group=f"identity:{group_token}",
                    identity_confirmed=group_confirmed,
                    source_video=item["source_video"],
                    track_id=item["track_id"],
                    vehicle_group_id=item["vehicle_group_id"],
                    identity_status="confirmed" if group_confirmed else "unknown_identity",
                    identity_evidence=item["identity_evidence"],
                    identity_subject=item["identity_subject"],
                )
            )

    samples.sort(key=lambda sample: str(sample.image_path).casefold())
    counts = Counter(sample.label for sample in samples)
    confirmed_groups = {sample.identity_group for sample in samples if sample.identity_confirmed}
    confirmed_by_label = {
        label: len({sample.identity_group for sample in samples if sample.identity_confirmed and sample.label == label})
        for label in TRAINING_LABELS
    }
    summary = {
        "manifest_rows": len(manifest),
        "eligible_rows_before_deduplication": len(raw),
        "usable_unique_rows": len(samples),
        "counts_by_label": {label: counts[label] for label in TRAINING_LABELS},
        "excluded_rows": len(excluded),
        "exact_duplicate_rows_removed": duplicate_rows,
        "conflicting_duplicate_rows_removed": conflict_rows,
        "confirmed_identity_groups": len(confirmed_groups),
        "confirmed_identity_groups_by_label": confirmed_by_label,
        "unconfirmed_identity_rows": sum(not sample.identity_confirmed for sample in samples),
        "validation_warning": (
            "Samples without a confirmed vehicle/session identity are reference-only and are not used as independent validation examples."
        ),
    }
    return AuditResult(tuple(samples), tuple(excluded), summary)
