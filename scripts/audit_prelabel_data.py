"""Audit reviewed images before they are used by the v0.3 pre-labeler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vehicleai.classification.data_audit import audit_human_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 VehicleAI v0.3 人工参考样本")
    parser.add_argument("--manifest", type=Path, default=Path("data/curated/curated_manifest.jsonl"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/annotations.jsonl"))
    parser.add_argument("--identities", type=Path, default=Path("data/annotations/vehicle_identities.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/prelabeling/data_audit.json"))
    parser.add_argument("--near-duplicate-distance", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_human_samples(
        args.manifest,
        args.annotations,
        near_duplicate_distance=args.near_duplicate_distance,
        identities_path=args.identities,
    )
    payload = {
        "summary": result.summary,
        "samples": [sample.to_dict() for sample in result.samples],
        "excluded": list(result.excluded),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    print(f"审计详情：{args.output.resolve()}")


if __name__ == "__main__":
    main()
