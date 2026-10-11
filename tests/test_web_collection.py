import hashlib
import io

import pytest
from PIL import Image

from vehicleai.collection.web import build_candidate_filename, find_near_duplicates, inspect_image_bytes


def image_bytes(size: tuple[int, int] = (800, 600), color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_filename_is_hash_based_stable_and_ascii() -> None:
    content = image_bytes()
    inspection = inspect_image_bytes(content, min_width=640, min_height=360, min_pixels=300_000)
    assert inspection.sha256 == hashlib.sha256(content).hexdigest()
    assert build_candidate_filename("web", "charger", inspection) == f"web_charger_{inspection.sha256[:12]}.jpg"


def test_low_resolution_and_corrupt_images_are_rejected() -> None:
    with pytest.raises(ValueError, match="too small"):
        inspect_image_bytes(image_bytes((320, 240)), min_width=640, min_height=360, min_pixels=300_000)
    with pytest.raises(ValueError, match="Corrupted"):
        inspect_image_bytes(b"not an image", min_width=1, min_height=1, min_pixels=1)


def test_near_duplicate_is_flagged() -> None:
    inspection = inspect_image_bytes(image_bytes(), min_width=1, min_height=1, min_pixels=1)
    assert find_near_duplicates(inspection.phash, {inspection.phash: "existing.jpg"}, 0) == ["existing.jpg"]
