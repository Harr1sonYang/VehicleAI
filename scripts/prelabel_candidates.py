"""Predict labels for unreviewed candidates without modifying annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

from vehicleai.classification.curation import AnnotationStore, discover_candidates
from vehicleai.classification.data_audit import audit_human_samples
from vehicleai.classification.embeddings import DinoV2Extractor, EmbeddingCache
from vehicleai.classification.prelabel import (
    BalancedKNNClassifier,
    prediction_record,
    write_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为未审核候选生成独立的 AI 预标注")
    parser.add_argument("--root", action="append", type=Path, dest="roots")
    parser.add_argument("--manifest", type=Path, default=Path("data/curated/curated_manifest.jsonl"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/annotations.jsonl"))
    parser.add_argument("--cache", type=Path, default=Path("data/embeddings/dinov2_vits14.npz"))
    parser.add_argument("--hub-cache", type=Path, default=Path("models/torch_hub"))
    parser.add_argument("--output", type=Path, default=Path("data/predictions/prelabels.jsonl"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--similarity-threshold", type=float, default=0.55)
    parser.add_argument("--margin-threshold", type=float, default=0.02)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = args.roots or [Path("data/raw/web"), Path("data/raw/video_candidates"), Path("data/candidates")]
    audit = audit_human_samples(args.manifest, args.annotations)
    annotations = AnnotationStore(args.annotations).latest
    candidates = {}
    for root in roots:
        if root.is_dir():
            for candidate in discover_candidates(root):
                if candidate.key not in annotations and candidate.candidate_status != "filtered":
                    candidates.setdefault(candidate.key, candidate)
    extractor = DinoV2Extractor(device=args.device, hub_cache=args.hub_cache, allow_download=not args.offline)
    cache = EmbeddingCache(args.cache, extractor.model_name)
    reference_vectors, _ = cache.embeddings_for(
        [sample.image_path for sample in audit.samples], extractor, batch_size=args.batch_size
    )
    classifier = BalancedKNNClassifier(
        k_per_class=args.k,
        similarity_threshold=args.similarity_threshold,
        margin_threshold=args.margin_threshold,
    ).fit(reference_vectors, [sample.label for sample in audit.samples], [sample.image_path for sample in audit.samples])
    candidate_list = list(candidates.values())
    if candidate_list:
        vectors, _ = cache.embeddings_for(
            [candidate.image_path for candidate in candidate_list], extractor, batch_size=args.batch_size
        )
        predictions = classifier.predict(vectors)
    else:
        predictions = []
    records = [
        prediction_record(candidate.image_path, prediction, model_name=extractor.model_name)
        for candidate, prediction in zip(candidate_list, predictions)
    ]
    count = write_predictions(records, args.output)
    print(f"未审核候选：{len(candidate_list)}；写入预测：{count}")
    print(f"预测文件：{args.output.resolve()}（人工标注未被修改）")


if __name__ == "__main__":
    main()
