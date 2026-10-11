"""Audit downloaded web candidates without modifying images or metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from vehicleai.collection.web import find_near_duplicates, inspect_image_bytes

REQUIRED_FIELDS = (
    "class_hint", "query", "source_page_url", "source_url", "artist",
    "license", "license_url", "sha256", "phash", "width", "height",
)
FILENAME_PATTERN = re.compile(r"web_(charger|explorer|durango|other)_[0-9a-f]{12}\.(jpg|png|webp)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/web"))
    parser.add_argument("--phash-distance", type=int, default=6)
    parser.add_argument("--file-list", type=Path, help="Audit only image paths listed in this UTF-8 text file")
    parser.add_argument("--contact-sheet-dir", type=Path, help="Optionally write one visual contact sheet per class hint")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allowed_paths = None
    if args.file_list:
        allowed_paths = {
            Path(line.lstrip("\ufeff")).resolve()
            for line in args.file_list.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        }
    records: list[tuple[dict, Path]] = []
    errors: list[str] = []
    for metadata_path in sorted(args.root.rglob("metadata.jsonl")):
        for line_number, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            image_path = metadata_path.parent / record["filename"]
            if allowed_paths is not None and image_path.resolve() not in allowed_paths:
                continue
            if not image_path.is_file():
                errors.append(f"{metadata_path}:{line_number}: missing image {image_path.name}")
                continue
            content = image_path.read_bytes()
            try:
                inspection = inspect_image_bytes(content, min_width=1, min_height=1, min_pixels=1)
            except ValueError as error:
                errors.append(f"{image_path}: {error}")
                continue
            missing = [field for field in REQUIRED_FIELDS if record.get(field) in (None, "")]
            checks = {
                "opens": True,
                "sha256_match": hashlib.sha256(content).hexdigest() == record.get("sha256") == inspection.sha256,
                "phash_match": inspection.phash == record.get("phash"),
                "dimensions_match": [inspection.width, inspection.height] == [record.get("width"), record.get("height")],
                "filename_pattern": bool(FILENAME_PATTERN.fullmatch(image_path.name)) and inspection.sha256[:12] in image_path.name,
                "missing_fields": missing,
            }
            errors.extend(f"{image_path}: failed {name}" for name, passed in checks.items() if name != "missing_fields" and not passed)
            errors.extend(f"{image_path}: missing {field}" for field in missing)
            records.append((record, image_path))
            print(json.dumps({
                "filename": image_path.name,
                "class_hint": record.get("class_hint"),
                "query": record.get("query"),
                "dimensions": f"{inspection.width}x{inspection.height}",
                "license": record.get("license"),
                "license_url": record.get("license_url"),
                "author": record.get("artist"),
                "source_page_url": record.get("source_page_url"),
                "near_duplicate_of": record.get("near_duplicate_of", []),
                "checks": checks,
            }, ensure_ascii=False))

    pairwise_flags: list[dict[str, object]] = []
    for index, (record, path) in enumerate(records):
        known = {str(other[0]["phash"]): other[1].name for other in records[index + 1:]}
        for match in find_near_duplicates(str(record["phash"]), known, args.phash_distance):
            pairwise_flags.append({"filename": path.name, "near_duplicate": match})

    if args.contact_sheet_dir:
        args.contact_sheet_dir.mkdir(parents=True, exist_ok=True)
        class_hints = sorted({str(record.get("class_hint")) for record, _ in records})
        for class_hint in class_hints:
            class_records = [(record, path) for record, path in records if record.get("class_hint") == class_hint]
            tile_width, tile_height, label_height, columns = 640, 420, 42, 2
            rows = math.ceil(len(class_records) / columns)
            sheet = Image.new("RGB", (tile_width * columns, (tile_height + label_height) * rows), "white")
            draw = ImageDraw.Draw(sheet)
            for position, (_, image_path) in enumerate(class_records):
                column, row = position % columns, position // columns
                with Image.open(image_path) as source:
                    tile = ImageOps.contain(source.convert("RGB"), (tile_width, tile_height))
                x = column * tile_width + (tile_width - tile.width) // 2
                y = row * (tile_height + label_height) + (tile_height - tile.height) // 2
                sheet.paste(tile, (x, y))
                draw.text((column * tile_width + 8, row * (tile_height + label_height) + tile_height + 8), image_path.name, fill="black")
            sheet.save(args.contact_sheet_dir / f"{class_hint}.jpg", quality=92)
    print(json.dumps({
        "summary": {
            "images": len(records),
            "errors": errors,
            "near_duplicate_flags": pairwise_flags,
            "passed": not errors,
        }
    }, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
