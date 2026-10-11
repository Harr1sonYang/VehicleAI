"""Interactively label VehicleAI candidate crops with keyboard controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from vehicleai.classification.curation import AnnotationStore, KEY_LABELS, Candidate, discover_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("data/candidates"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/annotations.jsonl"))
    parser.add_argument("--include-annotated", action="store_true", help="Show samples already labeled")
    return parser.parse_args()


def _display(candidate: Candidate, position: int, total: int):
    image = cv2.imread(str(candidate.image_path))
    if image is None:
        raise ValueError(f"Unable to decode image: {candidate.image_path}")
    max_width, max_height = 1280, 800
    scale = min(max_width / image.shape[1], max_height / image.shape[0], 1.0)
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = cv2.copyMakeBorder(image, 90, 10, 10, 10, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    lines = [
        f"{position}/{total}  {candidate.image_path.name}",
        "1 Charger | 2 Explorer | 3 Durango | 4 Other | 5 Unknown | S Skip | Q Save/Quit",
        f"hint={candidate.class_hint or '-'}  provider={candidate.provider or '-'}  frame={candidate.frame_number if candidate.frame_number is not None else '-'}",
    ]
    for line_index, text in enumerate(lines):
        cv2.putText(canvas, text, (15, 25 + line_index * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    args = parse_args()
    store = AnnotationStore(args.annotations)
    candidates = discover_candidates(args.candidates)
    if not args.include_annotated:
        candidates = [candidate for candidate in candidates if not store.is_annotated(candidate)]
    if not candidates:
        print("No unannotated candidate images found.")
        return 0

    window = "VehicleAI v0.2 Dataset Annotation"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        for position, candidate in enumerate(candidates, start=1):
            try:
                cv2.imshow(window, _display(candidate, position, len(candidates)))
            except ValueError as error:
                print(f"WARNING: {error}", file=sys.stderr)
                continue
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key in KEY_LABELS:
                    label = KEY_LABELS[key]
                    store.record(candidate, label)
                    print(f"[{position}/{len(candidates)}] {label}: {candidate.image_path}")
                    break
                if key in (ord("s"), ord("S")):
                    print(f"[{position}/{len(candidates)}] skipped: {candidate.image_path}")
                    break
                if key in (ord("q"), ord("Q")):
                    print(f"Saved annotations to {args.annotations}; quitting.")
                    return 0
    finally:
        cv2.destroyAllWindows()
    print(f"Annotation pass complete. Saved to {args.annotations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
