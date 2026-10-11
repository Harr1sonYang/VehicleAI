"""Plan, probe, or collect candidate images from documented public APIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from vehicleai.collection.web import (
    PROVIDER_NAME, WikimediaCommonsProvider, append_metadata,
    build_candidate_filename, find_near_duplicates, inspect_image_bytes,
    load_existing_hashes, utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/web_collection.yaml"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print the plan without network access (default)")
    mode.add_argument("--probe", action="store_true", help="Query a few metadata records without downloading images")
    mode.add_argument("--download", action="store_true", help="Download and validate candidates")
    parser.add_argument("--probe-limit", type=int, default=1, help="Metadata results per query in probe mode")
    parser.add_argument("--approve-provider", help="Required with --download; must equal wikimedia_commons")
    parser.add_argument("--max-per-class", type=int, help="Optional safety cap below configured target counts")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Collection config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("provider") != PROVIDER_NAME:
        raise ValueError(f"Unsupported provider: {config.get('provider')}")
    if not config.get("collections"):
        raise ValueError("No collections configured")
    return config


def plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "dry_run", "network_requests": 0, "downloads": 0,
        "provider": config["provider"], "api_endpoint": config["api_endpoint"],
        "output_root": config["output_root"],
        "filename_format": "web_<collection_hint>_<sha256_first_12>.<decoded_format>",
        "collections": [
            {"class_hint": hint, "filename_hint": settings["filename_hint"],
             "target_count": settings["target_count"], "queries": settings["queries"]}
            for hint, settings in config["collections"].items()
        ],
    }


def make_provider(config: dict[str, Any]) -> WikimediaCommonsProvider:
    return WikimediaCommonsProvider(
        config["api_endpoint"], config["user_agent"],
        delay_seconds=float(config["request_delay_seconds"]),
        timeout_seconds=float(config["request_timeout_seconds"]),
        thumbnail_width=int(config["thumbnail_width"]),
    )


def probe(config: dict[str, Any], limit: int) -> None:
    provider = make_provider(config)
    for class_hint, settings in config["collections"].items():
        for query in settings["queries"]:
            total_hits = provider.search_count(str(query))
            results = list(provider.search(str(query), max(1, limit)))
            print(json.dumps({
                "class_hint": class_hint, "query": query, "total_hits": total_hits,
                "results": [{"title": item.title, "source_page_url": item.source_page_url,
                             "source_url": item.source_url,
                             "license": item.provider_metadata.get("license"),
                             "license_url": item.provider_metadata.get("license_url")}
                            for item in results],
            }, ensure_ascii=False))


def collect(config: dict[str, Any], max_per_class: int | None) -> None:
    output_root = Path(config["output_root"])
    provider = make_provider(config)
    exact_hashes, perceptual_hashes = load_existing_hashes(output_root)
    seen_pages: set[int] = set()
    for class_hint, settings in config["collections"].items():
        target = int(settings["target_count"])
        if max_per_class is not None:
            target = min(target, max_per_class)
        class_dir = output_root / class_hint
        class_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = class_dir / "metadata.jsonl"
        accepted = 0
        for query in settings["queries"]:
            if accepted >= target:
                break
            for item in provider.search(str(query), max(25, (target - accepted) * 4)):
                if accepted >= target:
                    break
                if item.page_id in seen_pages:
                    continue
                seen_pages.add(item.page_id)
                try:
                    content = provider.download(item.download_url, int(config["max_download_bytes"]))
                    inspection = inspect_image_bytes(
                        content, min_width=int(config["min_width"]),
                        min_height=int(config["min_height"]), min_pixels=int(config["min_pixels"]),
                    )
                except (ValueError, OSError) as error:
                    print(f"REJECT {item.source_page_url}: {error}", file=sys.stderr)
                    continue
                if inspection.sha256 in exact_hashes:
                    print(f"DUPLICATE {item.source_page_url}: {exact_hashes[inspection.sha256]}")
                    continue
                filename = build_candidate_filename("web", str(settings["filename_hint"]), inspection)
                destination = class_dir / filename
                near_duplicates = find_near_duplicates(inspection.phash, perceptual_hashes, int(config["phash_distance"]))
                destination.write_bytes(content)
                record = {
                    "filename": filename, "image_path": filename, "class_hint": class_hint,
                    "label": None, "provider": PROVIDER_NAME, "query": query,
                    "source_url": item.source_url, "download_url": item.download_url,
                    "source_page_url": item.source_page_url, "provider_page_id": item.page_id,
                    "provider_title": item.title, "sha256": inspection.sha256,
                    "phash": inspection.phash, "near_duplicate_of": near_duplicates,
                    "width": inspection.width, "height": inspection.height,
                    "mime_type": inspection.mime_type, "collected_at": utc_now(),
                    **item.provider_metadata,
                }
                append_metadata(metadata_path, record)
                exact_hashes[inspection.sha256] = filename
                perceptual_hashes[inspection.phash] = filename
                accepted += 1
                marker = " NEAR_DUPLICATE" if near_duplicates else ""
                print(f"[{class_hint} {accepted}/{target}] {filename}{marker}")
        print(f"{class_hint}: accepted {accepted}/{target}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if not args.probe and not args.download:
        print(json.dumps(plan(config), ensure_ascii=False, indent=2))
        return 0
    if args.probe:
        if args.probe_limit < 1 or args.probe_limit > 5:
            raise ValueError("--probe-limit must be between 1 and 5")
        probe(config, args.probe_limit)
        return 0
    if args.approve_provider != PROVIDER_NAME:
        print(f"ERROR: downloads require --approve-provider {PROVIDER_NAME}. Run --dry-run and --probe first.", file=sys.stderr)
        return 2
    if args.max_per_class is not None and args.max_per_class < 1:
        raise ValueError("--max-per-class must be positive")
    collect(config, args.max_per_class)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
