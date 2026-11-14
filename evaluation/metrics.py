"""Metrics calculated only from supplied labels and model outputs."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    known = np.isfinite(labels) & np.isfinite(probabilities)
    y = labels[known].astype(int)
    p = probabilities[known]
    if not len(y):
        return {"sample_count": 0, "accuracy": None, "precision": None, "recall": None,
                "f1": None, "auroc": None, "auprc": None, "brier": None,
                "confusion_matrix": None}
    pred = (p >= 0.5).astype(int)
    both_classes = len(np.unique(y)) == 2
    expected_calibration_error = 0.0
    for bin_index, lower in enumerate(np.linspace(0.0, 0.9, 10)):
        upper_match = p <= lower + 0.1 if bin_index == 9 else p < lower + 0.1
        selected = (p >= lower) & upper_match
        if selected.any():
            expected_calibration_error += (
                float(selected.mean())
                * abs(float(y[selected].mean()) - float(p[selected].mean()))
            )
    roc_points = None
    pr_points = None
    if both_classes:
        fpr, tpr, _ = roc_curve(y, p)
        precision_values, recall_values, _ = precision_recall_curve(y, p)
        roc_points = {"false_positive_rate": fpr.tolist(), "true_positive_rate": tpr.tolist()}
        pr_points = {
            "precision": precision_values.tolist(),
            "recall": recall_values.tolist(),
        }
    return {
        "sample_count": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if both_classes else None,
        "auprc": float(average_precision_score(y, p)) if both_classes else None,
        "brier": float(brier_score_loss(y, p)),
        "expected_calibration_error_10_bins": float(expected_calibration_error),
        "roc_curve": roc_points,
        "precision_recall_curve": pr_points,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def multiclass_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(labels)
    y = labels[valid].astype(int)
    p = probabilities[valid]
    pred = p.argmax(axis=1) if len(y) else np.array([], dtype=int)
    class_count = probabilities.shape[1]
    all_classes_present = len(np.unique(y)) == class_count
    one_hot = np.eye(class_count)[y] if len(y) else np.empty((0, class_count))
    if len(y):
        confidence = p.max(axis=1)
        correct = (pred == y).astype(float)
        ece = 0.0
        for bin_index, lower in enumerate(np.linspace(0.0, 0.9, 10)):
            upper_match = (
                confidence <= lower + 0.1
                if bin_index == 9
                else confidence < lower + 0.1
            )
            selected = (confidence >= lower) & upper_match
            if selected.any():
                ece += float(selected.mean()) * abs(
                    float(correct[selected].mean()) - float(confidence[selected].mean())
                )
    else:
        ece = None
    return {
        "sample_count": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)) if len(y) else None,
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0))
        if len(y) else None,
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0))
        if len(y) else None,
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0))
        if len(y) else None,
        "confusion_matrix": confusion_matrix(
            y, pred, labels=np.arange(class_count)
        ).tolist() if len(y) else None,
        "auroc_ovr_macro": float(roc_auc_score(one_hot, p, average="macro"))
        if len(y) and all_classes_present else None,
        "auprc_macro": float(average_precision_score(one_hot, p, average="macro"))
        if len(y) and all_classes_present else None,
        "brier": float(np.mean(np.sum((p - np.eye(p.shape[1])[y]) ** 2, axis=1)))
        if len(y) else None,
        "expected_calibration_error_10_bins": ece,
    }
