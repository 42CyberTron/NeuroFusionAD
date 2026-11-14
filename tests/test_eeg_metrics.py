import numpy as np

from evaluation.eeg_metrics import participant_metrics


def test_participant_metrics_include_class_results_curves_and_uncertainty():
    labels = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.6, 0.2, 0.2],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.1, 0.8],
            [0.2, 0.1, 0.7],
        ]
    )
    metrics = participant_metrics(labels, probabilities, bootstrap_replicates=50)
    assert metrics["participant_count"] == 6
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["auroc_ovr_macro"] == 1.0
    assert metrics["per_class"]["FTD"]["support"] == 2
    assert metrics["roc_curves_ovr"]["AD"]["auroc"] == 1.0
    assert metrics["stratified_bootstrap_95_ci"]["macro_f1_95_ci"] == [1.0, 1.0]
