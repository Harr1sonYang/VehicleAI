"""Create a source-grouped four-class dataset from VehicleAI annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vehicleai.classification.curation import AnnotationStore
from vehicleai.classification.dataset import build_split_plan, materialise_plan, plan_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/annotations.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/dataset"))
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Show the plan without copying files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = AnnotationStore(args.annotations)
    plan = build_split_plan(store.latest.values(), tuple(args.ratios), args.seed)
    groups = {sample.group: sample.split for sample in plan}
    print(json.dumps({"samples": len(plan), "groups": len(groups), "summary": plan_summary(plan)}, indent=2))
    if args.dry_run:
        for sample in plan:
            print(f"{sample.split:5} {sample.assigned_class:15} {sample.group} -> {sample.image_path}")
        print("Dry run only; no files copied.")
        return 0
    copied = materialise_plan(plan, args.output)
    print(f"Copied {copied} sample(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
