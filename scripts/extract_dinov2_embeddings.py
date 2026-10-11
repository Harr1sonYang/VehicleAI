"""Incrementally cache DINOv2 embeddings for audited human references."""

from __future__ import annotations

import argparse
from pathlib import Path

from vehicleai.classification.data_audit import audit_human_samples
from vehicleai.classification.embeddings import DinoV2Extractor, EmbeddingCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取并缓存 DINOv2 ViT-S/14 图像特征")
    parser.add_argument("--manifest", type=Path, default=Path("data/curated/curated_manifest.jsonl"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/annotations.jsonl"))
    parser.add_argument("--cache", type=Path, default=Path("data/embeddings/dinov2_vits14.npz"))
    parser.add_argument("--hub-cache", type=Path, default=Path("models/torch_hub"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--offline", action="store_true", help="只使用已缓存模型，不访问网络")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_human_samples(args.manifest, args.annotations)
    extractor = DinoV2Extractor(device=args.device, hub_cache=args.hub_cache, allow_download=not args.offline)
    cache = EmbeddingCache(args.cache, extractor.model_name)
    _, stats = cache.embeddings_for(
        [sample.image_path for sample in audit.samples],
        extractor,
        batch_size=args.batch_size,
        progress=lambda current, total: print(f"新特征 {current}/{total}"),
    )
    print(f"设备：{extractor.device}")
    print(f"请求 {stats['requested']}，新计算 {stats['computed']}，缓存复用 {stats['reused']}")
    print(f"特征缓存：{args.cache.resolve()}")


if __name__ == "__main__":
    main()
