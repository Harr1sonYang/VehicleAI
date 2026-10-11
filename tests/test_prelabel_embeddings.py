from pathlib import Path

import numpy as np
from PIL import Image

from vehicleai.classification.embeddings import EmbeddingCache


class FakeExtractor:
    model_name = "fake-model"

    def __init__(self) -> None:
        self.calls: list[list[Path]] = []

    def extract(self, paths: list[Path]) -> np.ndarray:
        self.calls.append(paths)
        return np.asarray([[float(index + 1), 1.0, 0.0] for index, _ in enumerate(paths)], dtype=np.float32)


def make_image(path: Path, colour: tuple[int, int, int]) -> None:
    Image.new("RGB", (20, 20), colour).save(path)


def test_embedding_cache_is_normalised_content_addressed_and_incremental(tmp_path: Path) -> None:
    first, duplicate, added = tmp_path / "first.png", tmp_path / "duplicate.png", tmp_path / "added.png"
    make_image(first, (10, 20, 30))
    duplicate.write_bytes(first.read_bytes())
    make_image(added, (90, 80, 70))
    extractor = FakeExtractor()
    cache_path = tmp_path / "cache.npz"
    cache = EmbeddingCache(cache_path, extractor.model_name)

    vectors, stats = cache.embeddings_for([first, duplicate], extractor, batch_size=8)
    assert stats == {"requested": 2, "computed": 1, "reused": 1}
    assert len(extractor.calls) == 1
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert np.allclose(vectors[0], vectors[1])

    reopened = EmbeddingCache(cache_path, extractor.model_name)
    _, stats = reopened.embeddings_for([first, added], extractor, batch_size=8)
    assert stats == {"requested": 2, "computed": 1, "reused": 1}
    assert reopened.size == 2


def test_embedding_cache_rejects_a_different_model(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    make_image(image, (1, 2, 3))
    cache_path = tmp_path / "cache.npz"
    EmbeddingCache(cache_path, "one").embeddings_for([image], FakeExtractor())
    try:
        EmbeddingCache(cache_path, "two")
    except ValueError as error:
        assert "model mismatch" in str(error)
    else:
        raise AssertionError("expected model mismatch")
