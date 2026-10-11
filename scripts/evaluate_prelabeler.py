"""Run strict identity-group validation for the v0.3 pre-labeler."""

from __future__ import annotations

import argparse
from pathlib import Path

from vehicleai.classification.data_audit import audit_human_samples
from vehicleai.classification.embeddings import DinoV2Extractor, EmbeddingCache
from vehicleai.classification.evaluation import evaluate_split, write_evaluation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 VehicleAI v0.3 AI 预标注")
    parser.add_argument("--manifest", type=Path, default=Path("data/curated/curated_manifest.jsonl"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/annotations.jsonl"))
    parser.add_argument("--cache", type=Path, default=Path("data/embeddings/dinov2_vits14.npz"))
    parser.add_argument("--hub-cache", type=Path, default=Path("models/torch_hub"))
    parser.add_argument("--output-json", type=Path, default=Path("results/prelabeling/evaluation.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("results/prelabeling/evaluation.md"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--similarity-threshold", type=float, default=0.55)
    parser.add_argument("--margin-threshold", type=float, default=0.02)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_human_samples(args.manifest, args.annotations)
    extractor = DinoV2Extractor(device=args.device, hub_cache=args.hub_cache, allow_download=not args.offline)
    cache = EmbeddingCache(args.cache, extractor.model_name)
    embeddings, cache_stats = cache.embeddings_for(
        [sample.image_path for sample in audit.samples], extractor, batch_size=args.batch_size
    )
    evaluation = evaluate_split(
        list(audit.samples),
        embeddings,
        k_per_class=args.k,
        similarity_threshold=args.similarity_threshold,
        margin_threshold=args.margin_threshold,
    )
    report = {
        "model": extractor.model_name,
        "device": str(extractor.device),
        "cache": cache_stats,
        "settings": {
            "k_per_class": args.k,
            "similarity_threshold": args.similarity_threshold,
            "margin_threshold": args.margin_threshold,
        },
        "audit": audit.summary,
        "evaluation": evaluation,
        "reliability_note": (
            "三类目标车型样本没有可信 vehicle_id/vehicle_group_id/source 字段，无法证明照片来自独立车辆；"
            "为避免泄漏，它们只作为参考，没有进入验证集。因此这些类别的 Precision/Recall/F1 暂不可靠，不能据此宣称准确率。"
        ),
    }
    report["settings"]["selected_thresholds"] = evaluation.get("selected_thresholds")
    write_evaluation_report(report, args.output_json, args.output_markdown)
    print(f"评估 JSON：{args.output_json.resolve()}")
    print(f"评估报告：{args.output_markdown.resolve()}")
    print(f"严格验证样本：{evaluation['validation_samples']}；身份交叉：{len(evaluation['identity_overlap'])}")


if __name__ == "__main__":
    main()
