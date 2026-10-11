"""Build or summarize the metadata-only VehicleAI curated manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from vehicleai.classification.manifest import (
    build_manifest_from_annotations,
    manifest_summary,
    read_curated_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/annotations.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/curated/curated_manifest.jsonl"))
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Read the existing curated manifest and print counts without rebuilding it",
    )
    return parser.parse_args()


def print_summary(records: list[dict[str, object]]) -> None:
    for label, count in manifest_summary(records).items():
        print(f"{label}: {count}")


def main() -> int:
    args = parse_args()
    if args.summary_only:
        records = read_curated_manifest(args.output)
        print(f"Curated manifest: {args.output.resolve()}")
    else:
        records = build_manifest_from_annotations(args.annotations, args.output)
        print(f"Wrote {len(records)} curated record(s) to {args.output.resolve()}")
    print_summary(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
