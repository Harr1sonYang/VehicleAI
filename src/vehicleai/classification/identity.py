"""Append-only, evidence-aware vehicle identity management."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def normalise_path(path: str | Path) -> str:
    return Path(path).resolve().as_posix().casefold()


def _normalise_source(value: str | None) -> str:
    return str(value or "unknown").replace("\\", "/").casefold()


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class VehicleIdentity:
    vehicle_identity_id: str
    subject_key: str
    status: str
    evidence: str
    manually_confirmed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_identity(
    *,
    image_path: str | Path,
    source_video: str | None = None,
    track_id: int | str | None = None,
    vehicle_id: str | int | None = None,
    provider: str | None = None,
    source_id: str | int | None = None,
    source_identity_confirmed: bool = False,
) -> VehicleIdentity:
    """Infer only identities supported by explicit or same-track evidence.

    Automatic cross-track groups are deliberately not accepted here: a track
    association is useful evidence but is not proof of real-world identity.
    """
    if vehicle_id not in {None, ""}:
        subject = f"explicit-vehicle:{vehicle_id}"
        return VehicleIdentity(_stable_id("vehicle", subject), subject, "confirmed", "explicit_vehicle_id")
    if source_video and track_id is not None:
        subject = f"video-track:{_normalise_source(source_video)}:{track_id}"
        return VehicleIdentity(_stable_id("track", subject), subject, "confirmed", "same_video_track")
    if source_identity_confirmed and provider and source_id not in {None, ""}:
        subject = f"source-record:{str(provider).casefold()}:{source_id}"
        return VehicleIdentity(_stable_id("source", subject), subject, "confirmed", "confirmed_source_record")
    subject = f"image:{normalise_path(image_path)}"
    return VehicleIdentity(_stable_id("unknown", subject), subject, "unknown_identity", "insufficient_identity_evidence")


class VehicleIdentityStore:
    """Append-only manual identity decisions; latest decision per subject wins."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._overrides: dict[str, VehicleIdentity] = {}
        self._suggestions: dict[str, set[str]] = {}
        self._load()

    @staticmethod
    def image_subject(image_path: str | Path) -> str:
        return f"image:{normalise_path(image_path)}"

    def _apply(self, record: dict[str, Any]) -> None:
        subjects = [str(value) for value in record.get("subject_keys", [])]
        action = record.get("action")
        if action == "suspected_duplicate":
            for subject in subjects:
                self._suggestions.setdefault(subject, set()).update(value for value in subjects if value != subject)
            return
        identity = VehicleIdentity(
            vehicle_identity_id=str(record["vehicle_identity_id"]),
            subject_key="",
            status=str(record.get("status", "confirmed")),
            evidence=str(record.get("evidence", "manual_confirmation")),
            manually_confirmed=bool(record.get("manually_confirmed", True)),
        )
        for subject in subjects:
            self._overrides[subject] = VehicleIdentity(
                identity.vehicle_identity_id,
                subject,
                identity.status,
                identity.evidence,
                identity.manually_confirmed,
            )

    def _load(self) -> None:
        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    self._apply(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError(f"Invalid identity record at {self.path}:{line_number}: {error}") from error

    def resolve(self, inferred: VehicleIdentity, image_path: str | Path) -> VehicleIdentity:
        image_subject = self.image_subject(image_path)
        return self._overrides.get(image_subject, self._overrides.get(inferred.subject_key, inferred))

    def _append(self, record: dict[str, Any]) -> dict[str, Any]:
        complete = {**record, "timestamp": datetime.now(timezone.utc).isoformat()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(complete, ensure_ascii=False) + "\n")
            handle.flush()
        self._apply(complete)
        return complete

    def _subjects_for_id(self, identity_id: str) -> set[str]:
        return {subject for subject, identity in self._overrides.items() if identity.vehicle_identity_id == identity_id}

    def merge(self, identities: Iterable[VehicleIdentity], *, note: str | None = None) -> dict[str, Any]:
        materialised = list(identities)
        subjects = {identity.subject_key for identity in materialised}
        for identity in materialised:
            subjects.update(self._subjects_for_id(identity.vehicle_identity_id))
        if len(subjects) < 2:
            raise ValueError("At least two distinct identity subjects are required")
        token = "|".join(sorted(subjects))
        return self._append(
            {
                "action": "confirm_merge",
                "subject_keys": sorted(subjects),
                "vehicle_identity_id": _stable_id("manual_vehicle", token),
                "status": "confirmed",
                "evidence": "manual_confirmation",
                "manually_confirmed": True,
                "note": note,
            }
        )

    def split(self, inferred: VehicleIdentity, image_path: str | Path, *, whole_subject: bool = False) -> dict[str, Any]:
        subject = inferred.subject_key if whole_subject else self.image_subject(image_path)
        return self._append(
            {
                "action": "split",
                "subject_keys": [subject],
                "vehicle_identity_id": _stable_id("manual_split", f"{subject}:{datetime.now(timezone.utc).isoformat()}"),
                "status": "confirmed",
                "evidence": "manual_split",
                "manually_confirmed": True,
            }
        )

    def mark_unknown(self, image_path: str | Path) -> dict[str, Any]:
        subject = self.image_subject(image_path)
        return self._append(
            {
                "action": "mark_unknown",
                "subject_keys": [subject],
                "vehicle_identity_id": _stable_id("unknown", subject),
                "status": "unknown_identity",
                "evidence": "manual_unknown",
                "manually_confirmed": True,
            }
        )

    def suggest_duplicate(self, identities: Iterable[VehicleIdentity]) -> dict[str, Any]:
        subjects = sorted({identity.subject_key for identity in identities})
        if len(subjects) < 2:
            raise ValueError("At least two identity subjects are required")
        return self._append(
            {
                "action": "suspected_duplicate",
                "subject_keys": subjects,
                "vehicle_identity_id": "",
                "status": "suspected_duplicate",
                "evidence": "manual_suspicion_not_identity_confirmation",
                "manually_confirmed": False,
            }
        )

    def suggestions_for(self, identity: VehicleIdentity) -> tuple[str, ...]:
        return tuple(sorted(self._suggestions.get(identity.subject_key, set())))


def find_cross_source_suggestions(
    samples: list[Any],
    embeddings: Any,
    *,
    minimum_similarity: float = 0.92,
    max_pairs_per_sample: int = 1,
) -> list[dict[str, Any]]:
    """Propose, but never apply, high-similarity cross-source identity pairs."""
    import numpy as np

    vectors = np.asarray(embeddings, dtype=np.float32)
    suggestions: list[dict[str, Any]] = []
    used_pairs: set[tuple[int, int]] = set()
    for left_index, left in enumerate(samples):
        candidates: list[tuple[float, int]] = []
        left_source = _normalise_source(left.source_video or str(left.image_path.parent))
        for right_index in range(left_index + 1, len(samples)):
            right = samples[right_index]
            right_source = _normalise_source(right.source_video or str(right.image_path.parent))
            if left.label != right.label or left.identity_group == right.identity_group or left_source == right_source:
                continue
            similarity = float(vectors[left_index] @ vectors[right_index])
            if similarity >= minimum_similarity:
                candidates.append((similarity, right_index))
        for similarity, right_index in sorted(candidates, reverse=True)[:max_pairs_per_sample]:
            pair = (left_index, right_index)
            if pair in used_pairs:
                continue
            used_pairs.add(pair)
            right = samples[right_index]
            suggestions.append(
                {
                    "action": "suspected_duplicate",
                    "subject_keys": [left.identity_subject, right.identity_subject],
                    "vehicle_identity_id": "",
                    "status": "suspected_duplicate",
                    "evidence": "dinov2_cross_source_similarity_candidate_only",
                    "manually_confirmed": False,
                    "similarity": similarity,
                    "left_image_path": str(left.image_path),
                    "right_image_path": str(right.image_path),
                }
            )
    return suggestions
