"""Predict an independent image directory without adding it to references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vehicleai.classification.blind_test import (
    blind_prediction_record,
    discover_blind_images,
    evaluate_blind_records,
    load_trusted_labels,
    reject_reference_overlaps,
    write_blind_predictions,
)
from vehicleai.classification.data_audit import audit_human_samples
from vehicleai.classification.embeddings import DinoV2Extractor, EmbeddingCache
from vehicleai.classification.prelabel import BalancedKNNClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对独立目录执行 VehicleAI 批量盲测")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/curated/curated_manifest.jsonl"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/annotations.jsonl"))
    parser.add_argument("--identities", type=Path, default=Path("data/annotations/vehicle_identities.jsonl"))
    parser.add_argument("--cache", type=Path, default=Path("data/embeddings/dinov2_vits14.npz"))
    parser.add_argument("--hub-cache", type=Path, default=Path("models/torch_hub"))
    parser.add_argument("--output", type=Path, default=Path("data/predictions/blind_test.jsonl"))
    parser.add_argument("--evaluation", type=Path, default=Path("results/prelabeling/blind_test_evaluation.json"))
    parser.add_argument("--trusted-labels", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--thresholds-from", type=Path, default=Path("results/prelabeling/evaluation.json"))
    parser.add_argument("--similarity-threshold", type=float)
    parser.add_argument("--margin-threshold", type=float)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_human_samples(args.manifest, args.annotations, identities_path=args.identities)
    references = list(audit.samples)
    blind_paths, rejected = reject_reference_overlaps(
        discover_blind_images(args.input_dir),
        {sample.sha256 for sample in references},
        {sample.dhash for sample in references},
    )
    extractor = DinoV2Extractor(device=args.device, hub_cache=args.hub_cache, allow_download=not args.offline)
    cache = EmbeddingCache(args.cache, extractor.model_name)
    reference_vectors, _ = cache.embeddings_for(
        [sample.image_path for sample in references], extractor, batch_size=args.batch_size
    )
    calibrated = {}
    if args.thresholds_from.is_file():
        calibrated = json.loads(args.thresholds_from.read_text(encoding="utf-8")).get("evaluation", {}).get(
            "selected_thresholds", {}
        )
    similarity_threshold = (
        args.similarity_threshold
        if args.similarity_threshold is not None
        else float(calibrated.get("similarity_threshold", 0.55))
    )
    margin_threshold = (
        args.margin_threshold
        if args.margin_threshold is not None
        else float(calibrated.get("margin_threshold", 0.08))
    )
    classifier = BalancedKNNClassifier(
        k_per_class=args.k,
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
    ).fit(reference_vectors, [sample.label for sample in references], [sample.image_path for sample in references])
    if blind_paths:
        blind_vectors, _ = cache.embeddings_for(blind_paths, extractor, batch_size=args.batch_size)
        predictions = classifier.predict(blind_vectors)
    else:
        predictions = []
    records = [
        blind_prediction_record(path, prediction, model_name=extractor.model_name)
        for path, prediction in zip(blind_paths, predictions)
    ] + rejected
    write_blind_predictions(records, args.output)
    print(f"盲测图片：{len(blind_paths)}；参考重叠拒绝：{len(rejected)}")
    print(f"预测记录：{args.output.resolve()}")
    if args.trusted_labels:
        evaluation = evaluate_blind_records(records, load_trusted_labels(args.trusted_labels))
        args.evaluation.parent.mkdir(parents=True, exist_ok=True)
        args.evaluation.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"可信人工标签评估：{evaluation['trusted_evaluated_samples']} 条")
        print(f"评估结果：{args.evaluation.resolve()}")


if __name__ == "__main__":
    main()
