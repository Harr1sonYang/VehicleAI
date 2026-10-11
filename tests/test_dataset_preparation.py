from pathlib import Path

from vehicleai.classification.dataset import build_split_plan, materialise_plan


def test_same_source_never_leaks_across_splits() -> None:
    records = [
        {"image_path": f"frame_{index}.jpg", "source": "same-session.mp4", "assigned_class": "other"}
        for index in range(20)
    ]
    plan = build_split_plan(records, seed=7)
    assert len({sample.split for sample in plan}) == 1
    assert len({sample.group for sample in plan}) == 1


def test_unknown_is_excluded() -> None:
    records = [
        {"image_path": "good.jpg", "source": "a.mp4", "assigned_class": "charger_target"},
        {"image_path": "bad.jpg", "source": "b.mp4", "assigned_class": "unknown"},
    ]
    plan = build_split_plan(records)
    assert [sample.image_path.name for sample in plan] == ["good.jpg"]


def test_materialise_copies_without_modifying_source(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"original")
    records = [{"image_path": str(source), "source": "drive.mp4", "assigned_class": "other"}]
    plan = build_split_plan(records, ratios=(1.0, 0.0, 0.0))
    assert materialise_plan(plan, tmp_path / "dataset") == 1
    assert source.read_bytes() == b"original"
    copies = list((tmp_path / "dataset" / "train" / "other").glob("*.jpg"))
    assert len(copies) == 1 and copies[0].read_bytes() == b"original"
    for split in ("train", "val", "test"):
        for label in ("charger_target", "explorer_target", "durango_target", "other"):
            assert (tmp_path / "dataset" / split / label).is_dir()


def test_materialise_refuses_nonempty_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"source")
    output = tmp_path / "dataset"
    existing = output / "val" / "other" / "old.jpg"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"old")
    plan = build_split_plan(
        [{"image_path": str(source), "source": "drive.mp4", "assigned_class": "other"}],
        ratios=(1.0, 0.0, 0.0),
    )
    try:
        materialise_plan(plan, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected existing dataset images to be rejected")
