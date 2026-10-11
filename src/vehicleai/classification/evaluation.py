"""Identity-group-aware validation and metrics for the pre-labeler."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .data_audit import AuditedSample, TRAINING_LABELS
from .prelabel import BalancedKNNClassifier


def split_reference_validation(
    samples: list[AuditedSample], *, validation_fraction: float = 0.2
) -> tuple[list[int], list[int]]:
    """Hold out whole confirmed identity groups; unconfirmed rows remain reference-only."""
    confirmed_groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        if sample.identity_confirmed:
            confirmed_groups.setdefault(sample.identity_group, []).append(index)
    validation_groups: set[str] = set()
    by_label: dict[str, list[str]] = {label: [] for label in TRAINING_LABELS}
    for group, indices in confirmed_groups.items():
        labels = {samples[index].label for index in indices}
        if len(labels) == 1:
            by_label[next(iter(labels))].append(group)
    for groups in by_label.values():
        ordered = sorted(groups, key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
        if len(ordered) >= 2:
            count = max(1, round(len(ordered) * validation_fraction))
            validation_groups.update(ordered[: min(count, len(ordered) - 1)])
    validation = [index for index, sample in enumerate(samples) if sample.identity_group in validation_groups]
    reference = [index for index, sample in enumerate(samples) if sample.identity_group not in validation_groups]
    return reference, validation


def split_reference_calibration_validation(
    samples: list[AuditedSample], *, holdout_fraction: float = 0.2
) -> tuple[list[int], list[int], list[int]]:
    """Create disjoint reference, threshold-calibration, and final validation sets."""
    groups_by_label: dict[str, list[str]] = {label: [] for label in TRAINING_LABELS}
    group_indices: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        if sample.identity_confirmed:
            group_indices.setdefault(sample.identity_group, []).append(index)
    for group, indices in group_indices.items():
        labels = {samples[index].label for index in indices}
        if len(labels) == 1:
            groups_by_label[next(iter(labels))].append(group)
    calibration_groups: set[str] = set()
    validation_groups: set[str] = set()
    for groups in groups_by_label.values():
        ordered = sorted(groups, key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
        if len(ordered) >= 3:
            count = max(1, round(len(ordered) * holdout_fraction))
            count = min(count, (len(ordered) - 1) // 2)
            calibration_groups.update(ordered[:count])
            validation_groups.update(ordered[count : count * 2])
    calibration = [index for index, sample in enumerate(samples) if sample.identity_group in calibration_groups]
    validation = [index for index, sample in enumerate(samples) if sample.identity_group in validation_groups]
    held_out = calibration_groups | validation_groups
    reference = [index for index, sample in enumerate(samples) if sample.identity_group not in held_out]
    return reference, calibration, validation


def _select_thresholds(predictions: list[Any], truth: list[str]) -> tuple[float, float, dict[str, Any]]:
    """Choose thresholds on calibration groups, never on final validation groups."""
    similarity_grid = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
    margin_grid = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08)
    best: tuple[float, float, float] | None = None
    best_labels: list[str] = []
    for similarity_threshold in similarity_grid:
        for margin_threshold in margin_grid:
            labels = [
                prediction.top3[0]["label"]
                if prediction.similarity >= similarity_threshold and prediction.margin >= margin_threshold
                else "unknown"
                for prediction in predictions
            ]
            correct = sum(predicted == actual for predicted, actual in zip(labels, truth))
            wrong = sum(predicted not in {actual, "unknown"} for predicted, actual in zip(labels, truth))
            abstained = sum(predicted == "unknown" for predicted in labels)
            objective = correct - 2.0 * wrong - 0.25 * abstained
            candidate = (objective, similarity_threshold, margin_threshold)
            if best is None or candidate > best:
                best, best_labels = candidate, labels
    assert best is not None
    return best[1], best[2], classification_metrics(
        truth,
        best_labels,
        [[item["label"] for item in prediction.top3] for prediction in predictions],
    )


def classification_metrics(
    true_labels: Iterable[str],
    predicted_labels: Iterable[str],
    top3_labels: Iterable[Iterable[str]],
) -> dict[str, Any]:
    truth, predictions = list(true_labels), list(predicted_labels)
    top3 = [list(values) for values in top3_labels]
    if not (len(truth) == len(predictions) == len(top3)):
        raise ValueError("Metric inputs must have equal lengths")
    matrix_labels = list(TRAINING_LABELS) + ["unknown"]
    matrix = {actual: {predicted: 0 for predicted in matrix_labels} for actual in TRAINING_LABELS}
    for actual, predicted in zip(truth, predictions):
        if actual in matrix and predicted in matrix_labels:
            matrix[actual][predicted] += 1
    per_class: dict[str, Any] = {}
    for label in TRAINING_LABELS:
        tp = sum(actual == label and predicted == label for actual, predicted in zip(truth, predictions))
        fp = sum(actual != label and predicted == label for actual, predicted in zip(truth, predictions))
        fn = sum(actual == label and predicted != label for actual, predicted in zip(truth, predictions))
        support = sum(actual == label for actual in truth)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {
        "sample_count": len(truth),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "top3_accuracy": (sum(actual in guesses for actual, guesses in zip(truth, top3)) / len(truth)) if truth else None,
        "abstention_rate": (sum(predicted == "unknown" for predicted in predictions) / len(predictions)) if predictions else None,
        "predicted_counts": dict(Counter(predictions)),
    }


def evaluate_split(
    samples: list[AuditedSample],
    embeddings: np.ndarray,
    *,
    k_per_class: int,
    similarity_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    reference, calibration, validation = split_reference_calibration_validation(samples)
    report: dict[str, Any] = {
        "reference_samples": len(reference),
        "validation_samples": len(validation),
        "calibration_samples": len(calibration),
        "reference_groups": len({samples[index].identity_group for index in reference}),
        "validation_groups": len({samples[index].identity_group for index in validation}),
        "calibration_groups": len({samples[index].identity_group for index in calibration}),
        "identity_overlap": sorted(
            {samples[index].identity_group for index in reference}
            & {samples[index].identity_group for index in validation}
        ),
    }
    if not reference or not validation:
        report["metrics"] = classification_metrics([], [], [])
        report["errors"] = []
        return report
    classifier = BalancedKNNClassifier(
        k_per_class=k_per_class,
        similarity_threshold=-1.0,
        margin_threshold=-1.0,
    ).fit(
        embeddings[reference],
        [samples[index].label for index in reference],
        [samples[index].image_path for index in reference],
    )
    if calibration:
        calibration_predictions = classifier.predict(embeddings[calibration])
        similarity_threshold, margin_threshold, calibration_metrics = _select_thresholds(
            calibration_predictions, [samples[index].label for index in calibration]
        )
        report["calibration_metrics"] = calibration_metrics
    report["selected_thresholds"] = {
        "similarity_threshold": similarity_threshold,
        "margin_threshold": margin_threshold,
    }
    classifier.similarity_threshold = similarity_threshold
    classifier.margin_threshold = margin_threshold
    predictions = classifier.predict(embeddings[validation])
    truth = [samples[index].label for index in validation]
    report["metrics"] = classification_metrics(
        truth,
        [prediction.predicted_label for prediction in predictions],
        [[item["label"] for item in prediction.top3] for prediction in predictions],
    )
    report["errors"] = [
        {
            "image_path": str(samples[index].image_path),
            "true_label": samples[index].label,
            "predicted_label": prediction.predicted_label,
            "similarity": prediction.similarity,
            "margin": prediction.margin,
            "nearest_reference_path": prediction.nearest_reference_path,
        }
        for index, prediction in zip(validation, predictions)
        if prediction.predicted_label != samples[index].label
    ][:20]
    return report


def write_evaluation_report(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = report["evaluation"]["metrics"]
    lines = [
        "# VehicleAI v0.3 预标注评估",
        "",
        "> 相似度是余弦相似度，不是真实概率。AI 预测不覆盖人工标签。",
        "",
        "## 数据审计",
        "",
        f"- 可用去重样本：{report['audit']['usable_unique_rows']}",
        f"- 已确认身份组：{report['audit']['confirmed_identity_groups']}",
        f"- 身份未确认、仅作参考：{report['audit']['unconfirmed_identity_rows']}",
        f"- 验证样本：{report['evaluation']['validation_samples']}",
        f"- 阈值校准样本：{report['evaluation'].get('calibration_samples', 0)}",
        f"- 参考/验证身份交叉：{len(report['evaluation']['identity_overlap'])}",
        "",
        "## 每类指标",
        "",
        "| 类别 | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in TRAINING_LABELS:
        values = metrics["per_class"][label]
        fmt = lambda value: "不可靠/无样本" if value is None else f"{value:.3f}"
        lines.append(f"| {label} | {fmt(values['precision'])} | {fmt(values['recall'])} | {fmt(values['f1'])} | {values['support']} |")
    top3 = "不可靠/无样本" if metrics["top3_accuracy"] is None else f"{metrics['top3_accuracy']:.3f}"
    abstention = "不可靠/无样本" if metrics["abstention_rate"] is None else f"{metrics['abstention_rate']:.3f}"
    lines.extend(
        [
            "",
            f"- Top-3 Accuracy：{top3}",
            f"- Unknown / Abstention Rate：{abstention}",
            "",
            "## 可靠性说明",
            "",
            report["reliability_note"],
            "",
            "## 混淆矩阵",
            "",
            "```json",
            json.dumps(metrics["confusion_matrix"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## 典型错误案例",
            "",
        ]
    )
    if report["evaluation"]["errors"]:
        for error in report["evaluation"]["errors"]:
            lines.append(f"- `{error['true_label']}` → `{error['predicted_label']}`：`{error['image_path']}`（相似度 {error['similarity']:.3f}）")
    else:
        lines.append("- 当前严格身份隔离验证中没有可列出的错误，或没有足够验证样本。")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
