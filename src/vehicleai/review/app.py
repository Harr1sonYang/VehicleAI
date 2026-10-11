"""Flask application for local, durable candidate-image review."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import Image, UnidentifiedImageError

from vehicleai.classification.curation import (
    LABELS,
    AnnotationStore,
    Candidate,
    discover_candidates,
)
from vehicleai.classification.prelabel import read_predictions
from vehicleai.review.groups import TrackGroupStore

DEFAULT_DATASET_ROOTS = (
    Path("data/raw/web"),
    Path("data/raw/video_candidates"),
    Path("data/candidates"),
)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _public_http_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlparse(value)
    return value if parsed.scheme in {"http", "https"} and parsed.netloc else None


def _candidate_id(candidate: Candidate) -> str:
    return hashlib.sha256(candidate.key.encode("utf-8")).hexdigest()[:24]


def _dimensions(candidate: Candidate) -> tuple[int | None, int | None]:
    if candidate.width and candidate.height:
        return candidate.width, candidate.height
    try:
        with Image.open(candidate.image_path) as image:
            return image.size
    except (OSError, UnidentifiedImageError):
        return None, None


def _discover_all(roots: Iterable[Path]) -> tuple[list[Candidate], list[Path]]:
    candidates: dict[str, Candidate] = {}
    active_roots: list[Path] = []
    for root in roots:
        resolved_root = root.resolve()
        if not resolved_root.is_dir():
            continue
        active_roots.append(resolved_root)
        for candidate in discover_candidates(resolved_root):
            if candidate.image_path.is_file() and _is_within(candidate.image_path, resolved_root):
                candidates.setdefault(candidate.key, candidate)
    return sorted(candidates.values(), key=lambda item: item.key), active_roots


def create_app(
    dataset_roots: Iterable[Path] | None = None,
    annotations_path: Path = Path("data/annotations/annotations.jsonl"),
    track_groups_path: Path = Path("data/annotations/track_groups.jsonl"),
    predictions_path: Path = Path("data/predictions/prelabels.jsonl"),
) -> Flask:
    """Create the loopback-only review app; the runner controls the bind address."""
    app = Flask(__name__)
    roots = tuple(Path(root) for root in (dataset_roots or DEFAULT_DATASET_ROOTS))
    candidates, active_roots = _discover_all(roots)
    by_id = {_candidate_id(candidate): candidate for candidate in candidates}
    store = AnnotationStore(Path(annotations_path))
    group_store = TrackGroupStore(Path(track_groups_path))
    predictions = read_predictions(Path(predictions_path))
    write_lock = threading.Lock()

    app.config.update(
        CANDIDATES=candidates,
        CANDIDATES_BY_ID=by_id,
        DATASET_ROOTS=active_roots,
        ANNOTATION_STORE=store,
        TRACK_GROUP_STORE=group_store,
        AI_PREDICTIONS=predictions,
    )

    def filtered_candidates(*, include_status: bool = True) -> list[Candidate]:
        status = request.args.get("status", "unreviewed")
        class_hint = request.args.get("class_hint", "")
        final_label = request.args.get("label", "")
        provider = request.args.get("provider", "")
        candidate_status = request.args.get("candidate_status", "reviewable")
        sort_order = request.args.get("sort", "default")
        if status not in {"all", "unreviewed", "reviewed"}:
            abort(400, description="无效的审核状态筛选")
        if candidate_status not in {"", "all", "reviewable", "accepted", "suspected_false_positive", "filtered"}:
            abort(400, description="无效的候选质量筛选")
        if sort_order not in {"default", "uncertain_first"}:
            abort(400, description="无效的排序方式")

        result: list[Candidate] = []
        latest = store.latest
        for candidate in candidates:
            annotation = latest.get(candidate.key)
            reviewed = annotation is not None
            if include_status:
                if status == "reviewed" and not reviewed:
                    continue
                if status == "unreviewed" and reviewed:
                    continue
            if class_hint and candidate.class_hint != class_hint:
                continue
            if provider and candidate.provider != provider:
                continue
            effective_candidate_status = candidate.candidate_status or "accepted"
            if candidate_status == "reviewable" and effective_candidate_status == "filtered":
                continue
            if candidate_status not in {"", "all", "reviewable"} and effective_candidate_status != candidate_status:
                continue
            if final_label and (not annotation or annotation.get("label") != final_label):
                continue
            result.append(candidate)
        if sort_order == "uncertain_first":
            result.sort(
                key=lambda item: (
                    predictions.get(item.key, {}).get("ai_recommendation") != "unknown",
                    predictions.get(item.key, {}).get("margin", 1.0),
                    predictions.get(item.key, {}).get("similarity", 1.0),
                    item.key,
                )
            )
        return result

    def serialise(candidate: Candidate) -> dict[str, Any]:
        identifier = _candidate_id(candidate)
        annotation = store.latest.get(candidate.key)
        width, height = _dimensions(candidate)
        source_video = candidate.source_video or candidate.source
        effective_group = group_store.group_for(source_video, candidate.track_id, candidate.vehicle_group_id)
        effective_track_ids = sorted(
            set(candidate.group_track_ids)
            | {
                int(item.track_id)
                for item in candidates
                if item.track_id is not None
                and group_store.group_for(item.source_video or item.source, item.track_id, item.vehicle_group_id)
                == effective_group
            }
        ) if effective_group else list(candidate.group_track_ids)
        prediction = predictions.get(candidate.key)
        ai_reference_url = None
        if prediction and prediction.get("nearest_reference_path"):
            ai_reference_url = f"/ai-reference/{identifier}"
        return {
            "id": identifier,
            "filename": candidate.image_path.name,
            "candidate_path": str(candidate.image_path),
            "image_url": f"/candidate-image/{identifier}",
            "class_hint": candidate.class_hint,
            "provider": candidate.provider,
            "query": candidate.query,
            "source": candidate.source,
            "source_id": candidate.source_id,
            "source_video": candidate.source_video,
            "source_page_url": _public_http_url(candidate.source_page_url),
            "frame_number": candidate.frame_number,
            "track_id": candidate.track_id,
            "vehicle_group_id": effective_group,
            "group_track_ids": effective_track_ids,
            "association_status": candidate.association_status,
            "suspected_duplicate_track_ids": list(candidate.suspected_duplicate_track_ids),
            "candidate_status": candidate.candidate_status or "accepted",
            "candidate_filter_reasons": list(candidate.candidate_filter_reasons),
            "track_first_frame": candidate.track_first_frame,
            "track_last_frame": candidate.track_last_frame,
            "width": width,
            "height": height,
            "reviewed": annotation is not None,
            "label": annotation.get("label") if annotation else None,
            "timestamp": annotation.get("timestamp") if annotation else None,
            "ai_recommendation": prediction.get("ai_recommendation") if prediction else None,
            "ai_similarity": prediction.get("similarity") if prediction else None,
            "ai_margin": prediction.get("margin") if prediction else None,
            "ai_top3": prediction.get("top3", []) if prediction else [],
            "ai_score_semantics": prediction.get("score_semantics") if prediction else None,
            "ai_nearest_reference_path": prediction.get("nearest_reference_path") if prediction else None,
            "ai_nearest_reference_url": ai_reference_url,
        }

    @app.get("/")
    def index() -> str:
        return render_template("index.html", labels=LABELS)

    @app.get("/api/state")
    def state():
        filtered = filtered_candidates()
        status_broadened_pool = filtered_candidates(include_status=False)
        try:
            requested_index = int(request.args.get("index", "0"))
        except ValueError:
            abort(400, description="index 必须是整数")
        index = max(0, min(requested_index, max(0, len(filtered) - 1)))
        latest = store.latest
        reviewed_count = sum(candidate.key in latest for candidate in status_broadened_pool)
        global_reviewed_count = sum(candidate.key in latest for candidate in candidates)
        requested_id = request.args.get("candidate_id")
        requested_candidate = by_id.get(requested_id) if requested_id else None
        if requested_candidate in filtered:
            current_candidate = requested_candidate
        elif filtered:
            current_candidate = filtered[index]
        else:
            current_candidate = None

        previous_id = next_id = next_unreviewed_id = None
        fallback_previous_id = fallback_next_id = None
        if current_candidate is not None and len(filtered) > 1:
            pool_index = filtered.index(current_candidate)
            previous_id = _candidate_id(filtered[(pool_index - 1) % len(filtered)])
            next_id = _candidate_id(filtered[(pool_index + 1) % len(filtered)])
            for offset in range(1, len(filtered)):
                possible = filtered[(pool_index + offset) % len(filtered)]
                if possible.key not in latest:
                    next_unreviewed_id = _candidate_id(possible)
                    break
        if current_candidate is not None and len(status_broadened_pool) > 1:
            broad_index = status_broadened_pool.index(current_candidate)
            fallback_previous_id = _candidate_id(
                status_broadened_pool[(broad_index - 1) % len(status_broadened_pool)]
            )
            fallback_next_id = _candidate_id(
                status_broadened_pool[(broad_index + 1) % len(status_broadened_pool)]
            )

        preload_id = next_unreviewed_id or next_id or fallback_next_id
        next_image_url = f"/candidate-image/{preload_id}" if preload_id else None
        return jsonify(
            {
                "candidate": serialise(current_candidate) if current_candidate else None,
                "index": index,
                "position": filtered.index(current_candidate) + 1 if current_candidate else 0,
                "filtered_count": len(filtered),
                "navigation_count": len(filtered),
                "total_count": len(status_broadened_pool),
                "reviewed_count": reviewed_count,
                "unreviewed_count": len(status_broadened_pool) - reviewed_count,
                "global_total_count": len(candidates),
                "global_reviewed_count": global_reviewed_count,
                "global_unreviewed_count": len(candidates) - global_reviewed_count,
                "next_image_url": next_image_url,
                "navigation": {
                    "previous_id": previous_id,
                    "next_id": next_id,
                    "next_unreviewed_id": next_unreviewed_id,
                    "fallback_previous_id": fallback_previous_id,
                    "fallback_next_id": fallback_next_id,
                },
                "filters": {
                    "class_hints": sorted({item.class_hint for item in candidates if item.class_hint}),
                    "providers": sorted({item.provider for item in candidates if item.provider}),
                    "labels": list(LABELS),
                    "candidate_statuses": ["reviewable", "accepted", "suspected_false_positive", "filtered", "all"],
                },
            }
        )

    @app.post("/api/annotations")
    def annotate():
        payload = request.get_json(silent=True) or {}
        identifier = payload.get("candidate_id")
        label = payload.get("label")
        annotation_method = payload.get("annotation_method", "manual")
        candidate = by_id.get(identifier)
        if candidate is None:
            abort(404, description="候选图片不存在")
        if label not in LABELS:
            abort(400, description="无效标签")
        if annotation_method not in {"manual", "ai_accepted"}:
            abort(400, description="无效标注方式")
        if annotation_method == "ai_accepted" and candidate.key in store.latest:
            abort(409, description="AI 推荐不能覆盖现有人工标注")
        with write_lock:
            record = store.record(candidate, label, annotation_method=annotation_method)
        return jsonify({"ok": True, "label": record["label"], "timestamp": record["timestamp"]})

    @app.post("/api/track-groups")
    def update_track_group():
        payload = request.get_json(silent=True) or {}
        candidate = by_id.get(payload.get("candidate_id"))
        if candidate is None or candidate.track_id is None:
            abort(404, description="轨迹候选不存在")
        source_video = candidate.source_video or candidate.source
        if not source_video:
            abort(400, description="候选缺少来源视频")
        action = payload.get("action")
        try:
            if action == "merge":
                track_ids = [int(value) for value in payload.get("track_ids", [])]
                available_ids = {
                    int(item.track_id)
                    for item in candidates
                    if item.track_id is not None and (item.source_video or item.source) == source_video
                }
                if not set(track_ids).issubset(available_ids):
                    abort(400, description="只能合并同一视频中存在的轨迹")
                intervals: dict[int, tuple[int, int]] = {}
                for item in candidates:
                    if (
                        item.track_id is not None
                        and int(item.track_id) in track_ids
                        and (item.source_video or item.source) == source_video
                        and item.track_first_frame is not None
                        and item.track_last_frame is not None
                    ):
                        intervals[int(item.track_id)] = (item.track_first_frame, item.track_last_frame)
                for index, left_id in enumerate(track_ids):
                    for right_id in track_ids[index + 1 :]:
                        if left_id in intervals and right_id in intervals:
                            left, right = intervals[left_id], intervals[right_id]
                            if max(left[0], right[0]) <= min(left[1], right[1]):
                                abort(400, description="同一时间存在的轨迹不能合并")
                record = group_store.merge(source_video, track_ids)
            elif action == "split":
                record = group_store.split(source_video, int(candidate.track_id))
            else:
                abort(400, description="无效的轨迹组操作")
        except (TypeError, ValueError) as error:
            abort(400, description=str(error))
        return jsonify({"ok": True, **record})

    @app.get("/candidate-image/<identifier>")
    def candidate_image(identifier: str):
        candidate = by_id.get(identifier)
        if candidate is None or not candidate.image_path.is_file():
            abort(404)
        if not any(_is_within(candidate.image_path, root) for root in active_roots):
            abort(404)
        return send_file(candidate.image_path, conditional=True, max_age=0)

    @app.get("/ai-reference/<identifier>")
    def ai_reference(identifier: str):
        candidate = by_id.get(identifier)
        prediction = predictions.get(candidate.key) if candidate else None
        reference = Path(str(prediction.get("nearest_reference_path"))).resolve() if prediction and prediction.get("nearest_reference_path") else None
        if reference is None or not reference.is_file():
            abort(404)
        if not any(_is_within(reference, root) for root in active_roots):
            abort(404)
        return send_file(reference, conditional=True, max_age=0)

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "candidate_count": len(candidates)})

    return app
