"""DINOv2 feature extraction with an incremental content-addressed cache."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
from PIL import Image


class BatchExtractor(Protocol):
    model_name: str

    def extract(self, paths: list[Path]) -> np.ndarray: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def l2_normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


class EmbeddingCache:
    """A compact NPZ cache keyed by image bytes rather than file path."""

    def __init__(self, path: Path, model_name: str) -> None:
        self.path = Path(path)
        self.model_name = model_name
        self._vectors: dict[str, np.ndarray] = {}
        if self.path.is_file():
            with np.load(self.path, allow_pickle=False) as data:
                cached_model = str(data["model_name"].item())
                if cached_model != model_name:
                    raise ValueError(f"Embedding cache model mismatch: {cached_model} != {model_name}")
                self._vectors = {
                    str(key): np.asarray(vector, dtype=np.float32)
                    for key, vector in zip(data["hashes"].tolist(), data["embeddings"])
                }

    @property
    def size(self) -> int:
        return len(self._vectors)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self._vectors)
        vectors = np.stack([self._vectors[key] for key in keys]) if keys else np.empty((0, 0), dtype=np.float32)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    model_name=np.asarray(self.model_name),
                    hashes=np.asarray(keys),
                    embeddings=vectors,
                )
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def embeddings_for(
        self,
        paths: Iterable[Path],
        extractor: BatchExtractor,
        *,
        batch_size: int = 16,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, dict[str, int]]:
        ordered = [Path(path).resolve() for path in paths]
        hashes = [_sha256(path) for path in ordered]
        missing_by_hash: dict[str, Path] = {}
        for digest, path in zip(hashes, ordered):
            if digest not in self._vectors:
                missing_by_hash.setdefault(digest, path)
        missing = list(missing_by_hash.items())
        for offset in range(0, len(missing), batch_size):
            batch = missing[offset : offset + batch_size]
            vectors = l2_normalise(extractor.extract([path for _, path in batch]))
            if len(vectors) != len(batch):
                raise ValueError("Extractor returned the wrong number of embeddings")
            for (digest, _), vector in zip(batch, vectors):
                self._vectors[digest] = vector
            if progress:
                progress(min(offset + len(batch), len(missing)), len(missing))
        if missing:
            self._save()
        result = np.stack([self._vectors[digest] for digest in hashes]) if hashes else np.empty((0, 0), dtype=np.float32)
        return l2_normalise(result), {"requested": len(ordered), "computed": len(missing), "reused": len(ordered) - len(missing)}


class DinoV2Extractor:
    """Official DINOv2 ViT-S/14 backbone loaded through PyTorch Hub."""

    model_name = "dinov2_vits14"

    def __init__(
        self,
        *,
        device: str = "auto",
        hub_cache: Path = Path("models/torch_hub"),
        allow_download: bool = True,
    ) -> None:
        import torch
        from torchvision import transforms

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.hub_cache = Path(hub_cache).resolve()
        self.hub_cache.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(self.hub_cache))
        local_repositories = sorted(self.hub_cache.glob("facebookresearch_dinov2_*"))
        checkpoint = self.hub_cache / "checkpoints" / "dinov2_vits14_pretrain.pth"
        if not allow_download and (not local_repositories or not checkpoint.is_file()):
            raise FileNotFoundError(
                f"DINOv2 offline cache is incomplete under {self.hub_cache}; run once without --offline."
            )
        if local_repositories:
            repository, source = str(local_repositories[-1]), "local"
        elif allow_download:
            repository, source = "facebookresearch/dinov2", "github"
        else:  # Covered by the explicit cache check above.
            raise FileNotFoundError(f"DINOv2 repository is not cached under {self.hub_cache}")
        self.model = torch.hub.load(repository, self.model_name, source=source, trust_repo=True)
        self.model.eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def extract(self, paths: list[Path]) -> np.ndarray:
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(self.transform(image.convert("RGB")))
        batch = self.torch.stack(images).to(self.device)
        with self.torch.inference_mode():
            if self.device.type == "cuda":
                with self.torch.autocast(device_type="cuda", dtype=self.torch.float16):
                    output = self.model(batch)
            else:
                output = self.model(batch)
        return l2_normalise(output.float().cpu().numpy())
