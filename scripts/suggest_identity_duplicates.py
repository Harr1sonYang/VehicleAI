"""Write cross-source appearance suggestions without merging vehicle identities."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from vehicleai.classification.data_audit import audit_human_samples
from vehicleai.classification.embeddings import DinoV2Extractor, EmbeddingCache
from vehicleai.classification.identity import find_cross_source_suggestions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成跨来源疑似同车候选；不自动合并身份")
    parser.add_argument("--manifest", type=Path, default=Path("data/curated/curated_manifest.jsonl"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/annotations.jsonl"))
    parser.add_argument("--identities", type=Path, default=Path("data/annotations/vehicle_identities.jsonl"))
    parser.add_argument("--cache", type=Path, default=Path("data/embeddings/dinov2_vits14.npz"))
    parser.add_argument("--hub-cache", type=Path, default=Path("models/torch_hub"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations/identity_suggestions.jsonl"))
    parser.add_argument("--minimum-similarity", type=float, default=0.92)
    parser.add_argument("--max-pairs-per-sample", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_human_samples(args.manifest, args.annotations, identities_path=args.identities)
    extractor = DinoV2Extractor(device=args.device, hub_cache=args.hub_cache, allow_download=not args.offline)
    cache = EmbeddingCache(args.cache, extractor.model_name)
    embeddings, _ = cache.embeddings_for([sample.image_path for sample in audit.samples], extractor)
    suggestions = find_cross_source_suggestions(
        list(audit.samples),
        embeddings,
        minimum_similarity=args.minimum_similarity,
        max_pairs_per_sample=args.max_pairs_per_sample,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for suggestion in suggestions:
            handle.write(
                json.dumps(
                    {**suggestion, "timestamp": datetime.now(timezone.utc).isoformat()},
                    ensure_ascii=False,
                )
                + "\n"
            )
    temporary.replace(args.output)
    print(f"疑似跨来源重复候选：{len(suggestions)}")
    print("这些记录仅供人工审核，没有修改或合并 vehicle_identity。")
    print(f"候选文件：{args.output.resolve()}")


if __name__ == "__main__":
    main()
