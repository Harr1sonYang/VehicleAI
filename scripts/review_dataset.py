"""Start the local VehicleAI candidate-review website."""

from __future__ import annotations

import argparse
from pathlib import Path

from vehicleai.review import DEFAULT_DATASET_ROOTS, create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 VehicleAI 本地候选图片审核网页")
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        dest="roots",
        help="候选图片目录；可重复指定。默认自动检查项目内三个受支持目录。",
    )
    parser.add_argument(
        "--identities",
        type=Path,
        default=Path("data/annotations/vehicle_identities.jsonl"),
        help="追加式车辆身份确认记录路径",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("results/prelabeling/evaluation.json"),
        help="严格验证评估 JSON 路径",
    )
    parser.add_argument(
        "--identity-suggestions",
        type=Path,
        default=Path("data/annotations/identity_suggestions.jsonl"),
        help="跨来源疑似同车候选 JSONL 路径",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations/annotations.jsonl"),
        help="追加式 JSONL 标注记录路径",
    )
    parser.add_argument("--port", type=int, default=8000, help="本地端口（默认：8000）")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/predictions/prelabels.jsonl"),
        help="AI 预标注 JSONL 路径",
    )
    parser.add_argument(
        "--duplicates",
        type=Path,
        default=Path("data/road_collection/candidate_duplicates.jsonl"),
        help="跨视频精确／近重复索引",
    )
    parser.add_argument(
        "--collection-statistics",
        type=Path,
        default=Path("data/road_collection/statistics.json"),
        help="道路视频批处理统计 JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port 必须在 1 到 65535 之间")
    roots = args.roots or list(DEFAULT_DATASET_ROOTS)
    app = create_app(
        roots,
        args.annotations,
        predictions_path=args.predictions,
        identities_path=args.identities,
        evaluation_path=args.evaluation,
        identity_suggestions_path=args.identity_suggestions,
        duplicates_path=args.duplicates,
        collection_statistics_path=args.collection_statistics,
    )
    count = len(app.config["CANDIDATES"])
    print(f"已发现 {count} 张候选图片")
    print(f"审核地址：http://127.0.0.1:{args.port}")
    print(f"标注记录：{args.annotations.resolve()}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
