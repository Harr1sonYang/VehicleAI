import json
from pathlib import Path

from PIL import Image

from vehicleai.review import create_app


def make_candidate(
    root: Path,
    name: str,
    *,
    class_hint: str = "charger_target",
    provider: str = "test_provider",
    source_page_url: str = "https://example.test/source",
    **extra_metadata,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / name
    Image.new("RGB", (64, 48), color=(30, 60, 90)).save(image_path)
    metadata = {
        "filename": name,
        "image_path": name,
        "class_hint": class_hint,
        "provider": provider,
        "query": f"query for {name}",
        "source_page_url": source_page_url,
        "source_url": "https://example.test/image.jpg",
        "source_id": f"id-{name}",
        "width": 64,
        "height": 48,
        **extra_metadata,
    }
    with (root / "metadata.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata) + "\n")
    return image_path


def test_annotation_persists_relabels_and_resumes(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    make_candidate(root, "a.jpg")
    make_candidate(root, "b.jpg", class_hint="other")
    annotations = tmp_path / "annotations.jsonl"
    app = create_app([root], annotations)
    client = app.test_client()

    first = client.get("/api/state").get_json()["candidate"]
    assert client.post("/api/annotations", json={"candidate_id": first["id"], "label": "other"}).status_code == 200
    assert client.post("/api/annotations", json={"candidate_id": first["id"], "label": "unknown"}).status_code == 200

    records = [json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines()]
    assert [record["label"] for record in records] == ["other", "unknown"]
    assert records[-1]["class_hint"] == "charger_target"
    assert records[-1]["source_id"] == "id-a.jpg"

    resumed = create_app([root], annotations).test_client().get("/api/state").get_json()
    assert resumed["reviewed_count"] == 1
    assert resumed["candidate"]["filename"] == "b.jpg"


def test_filters_and_multi_root_discovery(tmp_path: Path) -> None:
    web_root = tmp_path / "web"
    video_root = tmp_path / "video"
    make_candidate(web_root, "web.jpg", class_hint="charger_target", provider="commons")
    make_candidate(video_root, "video.jpg", class_hint="other", provider="local_video")
    annotations = tmp_path / "annotations.jsonl"
    client = create_app([web_root, video_root], annotations).test_client()

    state = client.get("/api/state?status=all&provider=local_video").get_json()
    assert state["total_count"] == 1
    assert state["global_total_count"] == 2
    assert state["filtered_count"] == 1
    assert state["candidate"]["filename"] == "video.jpg"
    identifier = state["candidate"]["id"]
    client.post("/api/annotations", json={"candidate_id": identifier, "label": "unusable"})
    labelled = client.get("/api/state?status=reviewed&label=unusable").get_json()
    assert labelled["filtered_count"] == 1
    assert labelled["candidate"]["label"] == "unusable"


def test_navigation_keeps_filters_strict_and_exposes_explicit_status_fallback(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    make_candidate(root, "a.jpg")
    make_candidate(root, "b.jpg")
    make_candidate(root, "c.jpg")
    annotations = tmp_path / "annotations.jsonl"
    client = create_app([root], annotations).test_client()

    all_state = client.get("/api/state?status=all").get_json()
    first_id = all_state["candidate"]["id"]
    second_id = all_state["navigation"]["next_id"]
    second = client.get(f"/api/state?status=all&candidate_id={second_id}").get_json()
    third_id = second["navigation"]["next_id"]
    client.post("/api/annotations", json={"candidate_id": second_id, "label": "other"})
    client.post("/api/annotations", json={"candidate_id": third_id, "label": "unknown"})

    remaining = client.get("/api/state?status=unreviewed").get_json()
    assert remaining["candidate"]["id"] == first_id
    assert remaining["filtered_count"] == 1
    assert remaining["navigation"]["previous_id"] is None
    assert remaining["navigation"]["next_id"] is None
    assert remaining["navigation"]["next_unreviewed_id"] is None
    assert remaining["navigation"]["fallback_previous_id"] == third_id
    assert remaining["navigation"]["fallback_next_id"] == second_id

    still_strict = client.get(f"/api/state?status=unreviewed&candidate_id={third_id}").get_json()
    assert still_strict["candidate"]["filename"] == "a.jpg"
    previous = client.get(f"/api/state?status=all&candidate_id={third_id}").get_json()
    assert previous["candidate"]["filename"] == "c.jpg"

    client.post("/api/annotations", json={"candidate_id": first_id, "label": "unusable"})
    completed = client.get("/api/state?status=unreviewed").get_json()
    assert completed["filtered_count"] == 0
    assert completed["candidate"] is None
    assert completed["reviewed_count"] == 3


def test_annotation_api_validation_and_opaque_image_access(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    make_candidate(root, "candidate.jpg")
    client = create_app([root], tmp_path / "annotations.jsonl").test_client()
    candidate = client.get("/api/state").get_json()["candidate"]

    assert client.get(candidate["image_url"]).status_code == 200
    assert client.get("/candidate-image/not-a-real-candidate").status_code == 404
    assert client.get("/candidate-image/..%2F..%2Fsecret.txt").status_code == 404
    assert client.post("/api/annotations", json={"candidate_id": candidate["id"], "label": "invalid"}).status_code == 400
    assert client.post("/api/annotations", json={"candidate_id": "missing", "label": "other"}).status_code == 404


def test_frontend_is_chinese_and_exposes_keyboard_controls(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    make_candidate(root, "candidate.jpg")
    client = create_app([root], tmp_path / "annotations.jsonl").test_client()

    page = client.get("/").get_data(as_text=True)
    script = client.get("/static/review.js").get_data(as_text=True)
    assert "VehicleAI 数据集审核" in page
    assert "不代表警用或执法用途" in page
    assert "data-label=\"unusable\"" in page
    for key in ("Digit1", "Numpad1", "Digit6", "Numpad6", "KeyS", "ArrowLeft", "ArrowRight"):
        assert key in script
    assert "event.preventDefault()" in script
    assert "}, true);" in script


def test_non_http_source_link_is_not_exposed(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    make_candidate(root, "candidate.jpg", source_page_url="javascript:alert(1)")
    state = create_app([root], tmp_path / "annotations.jsonl").test_client().get("/api/state").get_json()
    assert state["candidate"]["source_page_url"] is None


def test_track_group_merge_and_split_are_persisted(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    common = {"source": "drive.mp4", "source_video": "drive.mp4", "candidate_status": "accepted"}
    make_candidate(root, "track1.jpg", track_id=1, vehicle_group_id="vehicle_group_000001", **common)
    make_candidate(root, "track2.jpg", track_id=2, vehicle_group_id="vehicle_group_000002", **common)
    groups = tmp_path / "track_groups.jsonl"
    client = create_app([root], tmp_path / "annotations.jsonl", groups).test_client()
    first = client.get("/api/state?status=all").get_json()["candidate"]
    merged = client.post(
        "/api/track-groups",
        json={"candidate_id": first["id"], "action": "merge", "track_ids": [1, 2]},
    )
    assert merged.status_code == 200
    state = client.get(f"/api/state?status=all&candidate_id={first['id']}").get_json()["candidate"]
    assert state["group_track_ids"] == [1, 2]
    split = client.post(
        "/api/track-groups", json={"candidate_id": first["id"], "action": "split"}
    )
    assert split.status_code == 200
    state = client.get(f"/api/state?status=all&candidate_id={first['id']}").get_json()["candidate"]
    assert state["group_track_ids"] == [1]
    assert len(groups.read_text(encoding="utf-8").splitlines()) == 2


def test_filtered_candidates_are_hidden_by_default_but_reviewable_explicitly(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    make_candidate(root, "accepted.jpg", candidate_status="accepted")
    make_candidate(root, "filtered.jpg", candidate_status="filtered")
    client = create_app([root], tmp_path / "annotations.jsonl").test_client()
    default_state = client.get("/api/state?status=all").get_json()
    assert default_state["filtered_count"] == 1
    accepted_id = default_state["candidate"]["id"]
    filtered = client.get("/api/state?status=all&candidate_status=filtered").get_json()
    assert filtered["filtered_count"] == 1
    assert filtered["candidate"]["filename"] == "filtered.jpg"

    client.post("/api/annotations", json={"candidate_id": accepted_id, "label": "other"})
    no_reviewable_work = client.get("/api/state?status=unreviewed").get_json()
    assert no_reviewable_work["candidate"] is None
    assert no_reviewable_work["total_count"] == 1
    assert no_reviewable_work["reviewed_count"] == 1
    assert no_reviewable_work["unreviewed_count"] == 0
    assert no_reviewable_work["global_total_count"] == 2
    assert no_reviewable_work["global_unreviewed_count"] == 1
    all_quality = client.get("/api/state?status=unreviewed&candidate_status=all").get_json()
    assert all_quality["candidate"]["filename"] == "filtered.jpg"


def test_legacy_candidate_without_quality_status_is_treated_as_accepted(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    make_candidate(root, "legacy.jpg")
    client = create_app([root], tmp_path / "annotations.jsonl").test_client()
    state = client.get("/api/state?status=all&candidate_status=accepted").get_json()
    assert state["filtered_count"] == 1
    assert state["candidate"]["filename"] == "legacy.jpg"


def test_manual_merge_rejects_tracks_with_overlapping_frame_ranges(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    common = {"source": "drive.mp4", "source_video": "drive.mp4"}
    make_candidate(
        root,
        "track1.jpg",
        track_id=1,
        track_statistics={"first_frame": 10, "last_frame": 20},
        **common,
    )
    make_candidate(
        root,
        "track2.jpg",
        track_id=2,
        track_statistics={"first_frame": 15, "last_frame": 25},
        **common,
    )
    client = create_app([root], tmp_path / "annotations.jsonl", tmp_path / "groups.jsonl").test_client()
    candidate = client.get("/api/state?status=all").get_json()["candidate"]
    response = client.post(
        "/api/track-groups",
        json={"candidate_id": candidate["id"], "action": "merge", "track_ids": [1, 2]},
    )
    assert response.status_code == 400
    assert not (tmp_path / "groups.jsonl").exists()


def test_ai_prediction_is_separate_visible_and_accept_requires_explicit_action(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    image_path = make_candidate(root, "candidate.jpg")
    reference_path = make_candidate(root, "reference.jpg", class_hint="explorer_target")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "image_path": str(image_path.resolve()),
                "ai_recommendation": "explorer_target",
                "similarity": 0.812,
                "margin": 0.11,
                "top3": [
                    {"label": "explorer_target", "score": 0.812, "nearest_reference_path": str(reference_path)},
                    {"label": "other", "score": 0.702, "nearest_reference_path": str(reference_path)},
                ],
                "nearest_reference_path": str(reference_path.resolve()),
                "score_semantics": "cosine similarity; not a probability",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    annotations = tmp_path / "annotations.jsonl"
    client = create_app([root], annotations, predictions_path=predictions).test_client()
    state = client.get("/api/state?candidate_id=" + client.get("/api/state").get_json()["candidate"]["id"]).get_json()
    candidate = state["candidate"]
    assert candidate["ai_recommendation"] == "explorer_target"
    assert candidate["label"] is None
    assert not annotations.exists()
    assert client.get(candidate["ai_nearest_reference_url"]).status_code == 200
    response = client.post("/api/annotations", json={"candidate_id": candidate["id"], "label": candidate["ai_recommendation"]})
    assert response.status_code == 200
    assert json.loads(annotations.read_text(encoding="utf-8"))["label"] == "explorer_target"
    blocked = client.post(
        "/api/annotations",
        json={
            "candidate_id": candidate["id"],
            "label": "other",
            "annotation_method": "ai_accepted",
        },
    )
    assert blocked.status_code == 409
    assert len(annotations.read_text(encoding="utf-8").splitlines()) == 1
