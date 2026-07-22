"""Versioned product evaluation metrics from per-sample predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from rightarm_mlops.batch import LABELS
from rightarm_mlops.protocol import atomic_json


def evaluate_predictions(
    predictions_path: Path,
    output_path: Path,
    *,
    champion_evaluation: Optional[Path] = None,
) -> Dict[str, Any]:
    samples = list(_read_jsonl(predictions_path))
    if not samples:
        raise ValueError("predictions file is empty")
    labels = sorted(LABELS)
    confusion = {truth: {prediction: 0 for prediction in labels} for truth in labels}
    for line, sample in enumerate(samples, 1):
        truth = sample.get("truth")
        prediction = sample.get("prediction")
        if truth not in LABELS or prediction not in LABELS:
            raise ValueError(
                f"prediction row {line} contains an unknown truth/prediction label"
            )
        confusion[truth][prediction] += 1
    per_class = {}
    f1_values = []
    recalls = []
    for label in labels:
        tp = confusion[label][label]
        false_negative = sum(confusion[label].values()) - tp
        false_positive = sum(confusion[truth][label] for truth in labels) - tp
        precision = _divide(tp, tp + false_positive)
        recall = _divide(tp, tp + false_negative)
        f1 = _divide(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + false_negative,
        }
        f1_values.append(f1)
        recalls.append(recall)
    none_total = sum(confusion["NONE"].values())
    none_false_accept = sum(
        count for label, count in confusion["NONE"].items() if label != "NONE"
    )
    correct = sum(confusion[label][label] for label in labels)
    metrics = {
        "offline.accuracy": _divide(correct, len(samples)),
        "offline.macro_f1": sum(f1_values) / len(f1_values),
        "offline.min_per_class_recall": min(recalls),
        "offline.none_false_accept_rate": _divide(none_false_accept, none_total),
        "offline.sample_count": len(samples),
    }
    champion = None
    if champion_evaluation is not None:
        champion = json.loads(champion_evaluation.read_text(encoding="utf-8"))
        champion_metrics = champion["metrics"]
        metrics["offline.macro_f1_delta_vs_champion"] = (
            metrics["offline.macro_f1"] - champion_metrics["offline.macro_f1"]
        )
        metrics["offline.min_per_class_recall_delta_vs_champion"] = (
            metrics["offline.min_per_class_recall"]
            - champion_metrics["offline.min_per_class_recall"]
        )
    evaluation = {
        "schema_version": "rightarm.gesture.evaluation/v1",
        "labels": labels,
        "metrics": metrics,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "champion_evaluation": str(champion_evaluation.resolve())
        if champion_evaluation
        else None,
    }
    atomic_json(output_path, evaluation)
    return evaluation


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON on prediction row {line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(f"prediction row {line_number} is not an object")
                yield value


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
