"""Participant-level metrics for the ds004504 three-class classification task."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from datasets.openneuro import CLASS_NAMES


def _calibration(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == labels).astype(float)
    ece = 0.0
    bins: list[dict[str, float | int]] = []
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        selected = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= 1)
        if selected.any():
            mean_confidence = float(confidence[selected].mean())
            accuracy = float(correct[selected].mean())
            ece += float(selected.mean()) * abs(accuracy - mean_confidence)
            bins.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": int(selected.sum()),
                    "confidence": mean_confidence,
                    "accuracy": accuracy,
                }
            )
    one_hot = np.eye(len(CLASS_NAMES), dtype=float)[labels]
    return {
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "expected_calibration_error_10_bins": float(ece),
        "reliability_bins": bins,
    }


def _stratified_bootstrap(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    seed: int = 42,
    replicates: int = 2000,
) -> dict[str, list[float] | None]:
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(labels == index) for index in range(len(CLASS_NAMES))]
    if any(len(indices) < 2 for indices in by_class):
        return {"accuracy_95_ci": None, "balanced_accuracy_95_ci": None, "macro_f1_95_ci": None}

    scores = {"accuracy": [], "balanced_accuracy": [], "macro_f1": []}
    for _ in range(replicates):
        sample = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in by_class]
        )
        truth = labels[sample]
        predicted = predictions[sample]
        scores["accuracy"].append(float(accuracy_score(truth, predicted)))
        scores["balanced_accuracy"].append(float(balanced_accuracy_score(truth, predicted)))
        scores["macro_f1"].append(
            float(f1_score(truth, predicted, average="macro", zero_division=0))
        )
    return {
        f"{name}_95_ci": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        for name, values in scores.items()
    }


def participant_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.shape != (len(y), len(CLASS_NAMES)):
        raise ValueError("Expected labels [participants] and probabilities [participants, 3].")
    if not len(y) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Metrics require finite, non-empty participant probabilities.")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Each participant's class probabilities must sum to one.")
    if not set(np.unique(y)).issubset(set(range(len(CLASS_NAMES)))):
        raise ValueError("Labels are outside the configured class indices.")

    predicted = p.argmax(axis=1)
    present_classes = np.unique(y)
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.int64)[y]
    per_class: dict[str, Any] = {}
    roc_curves: dict[str, Any] = {}
    precision_recall_curves: dict[str, Any] = {}
    for index, name in enumerate(CLASS_NAMES):
        binary_truth = one_hot[:, index]
        per_class[name] = {
            "support": int(binary_truth.sum()),
            "precision": float(
                precision_score(y == index, predicted == index, zero_division=0)
            ),
            "recall": float(recall_score(y == index, predicted == index, zero_division=0)),
            "f1": float(f1_score(y == index, predicted == index, zero_division=0)),
        }
        if binary_truth.any() and (~binary_truth.astype(bool)).any():
            false_positive_rate, true_positive_rate, _ = roc_curve(binary_truth, p[:, index])
            precision, recall, _ = precision_recall_curve(binary_truth, p[:, index])
            roc_curves[name] = {
                "false_positive_rate": false_positive_rate.tolist(),
                "true_positive_rate": true_positive_rate.tolist(),
                "auroc": float(roc_auc_score(binary_truth, p[:, index])),
            }
            precision_recall_curves[name] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "auprc": float(average_precision_score(binary_truth, p[:, index])),
            }

    return {
        "participant_count": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision_macro": float(precision_score(y, predicted, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, predicted, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=range(len(CLASS_NAMES))).tolist(),
        "class_names": list(CLASS_NAMES),
        "per_class": per_class,
        "auroc_ovr_macro": (
            float(roc_auc_score(one_hot, p, average="macro", multi_class="ovr"))
            if len(present_classes) == len(CLASS_NAMES)
            else None
        ),
        "auprc_macro": (
            float(average_precision_score(one_hot, p, average="macro"))
            if len(present_classes) == len(CLASS_NAMES)
            else None
        ),
        "roc_curves_ovr": roc_curves,
        "precision_recall_curves_ovr": precision_recall_curves,
        "calibration": _calibration(p, y),
        "stratified_bootstrap_95_ci": _stratified_bootstrap(
            y, predicted, replicates=bootstrap_replicates
        ),
        "interpretation_note": (
            "Metrics use one mean-window probability vector per held-out participant. "
            "Bootstrap intervals resample participants within class and do not replace "
            "external validation."
        ),
    }
