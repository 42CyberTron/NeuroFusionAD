import numpy as np

from evaluation.metrics import binary_metrics, multiclass_metrics


def test_binary_metrics_return_real_calculations_and_sample_count():
    metrics = binary_metrics(
        np.array([0, 1, 0, 1], dtype=float),
        np.array([0.1, 0.9, 0.3, 0.7]),
    )
    assert metrics["sample_count"] == 4
    assert metrics["accuracy"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["brier"] < 0.1


def test_binary_metrics_report_undefined_auc_for_single_class():
    metrics = binary_metrics(np.array([1.0, 1.0]), np.array([0.6, 0.7]))
    assert metrics["auroc"] is None
    assert metrics["auprc"] is None


def test_multiclass_metrics_return_macro_auc_when_all_classes_are_present():
    metrics = multiclass_metrics(
        np.array([0, 1, 2]),
        np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["auroc_ovr_macro"] == 1.0
    assert metrics["auprc_macro"] == 1.0
