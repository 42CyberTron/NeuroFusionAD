"""Evaluate saved checkpoints on their untouched held-out participants."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from datasets.openneuro import CLASS_NAMES, load_dataset_index
from evaluation.eeg_metrics import participant_metrics
from models.eeg_network import EEGClinicalNet
from training.eeg_data import EEGSubject, load_subjects
from training.eeg_train import _participant_probabilities


MODEL_FILES = {
    "multimodal": "eeg_multimodal.pt",
    "eeg_only": "eeg_only.pt",
    "clinical_only": "clinical_only.pt",
}


def evaluate_checkpoints(
    data_root: str | Path = "data/openneuro_ds004504",
    cache_dir: str | Path = "data/processed/ds004504",
    checkpoint_dir: str | Path = "checkpoints",
    output_path: str | Path = "evaluation/metrics.json",
    *,
    device_name: str | None = None,
) -> dict[str, Any]:
    root = Path(data_root)
    cache = Path(cache_dir)
    _, raw_inventory = load_dataset_index(root)
    metadata_bytes = (
        (root / "participants.tsv").read_bytes()
        + (root / "dataset_description.json").read_bytes()
    )
    metadata_hash = hashlib.sha256(metadata_bytes).hexdigest()
    checkpoints: dict[str, dict[str, Any]] = {}
    for mode, name in MODEL_FILES.items():
        path = Path(checkpoint_dir) / name
        if not path.is_file():
            raise FileNotFoundError(f"Trained {mode} checkpoint is missing: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not checkpoint.get("trained") or checkpoint.get("mode") != mode:
            raise ValueError(f"{path} is not a trained {mode} checkpoint.")
        if checkpoint.get("cohort_metadata_sha256") != metadata_hash:
            raise ValueError("Current participant metadata differs from the training cohort.")
        checkpoints[mode] = checkpoint

    reference_split = checkpoints["multimodal"]["split_patient_ids"]
    if any(checkpoint["split_patient_ids"] != reference_split for checkpoint in checkpoints.values()):
        raise ValueError("Model checkpoints do not share the same participant split.")
    test_ids = reference_split.get("test")
    if not test_ids:
        raise ValueError("Checkpoint does not contain a held-out participant test set.")
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("Held-out participant list contains duplicates.")

    frame, _ = load_dataset_index(root)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    results: dict[str, Any] = {}
    prediction_records: dict[str, Any] = {}
    y_true: np.ndarray | None = None
    for mode, checkpoint in checkpoints.items():
        subjects: list[EEGSubject] = load_subjects(
            frame,
            test_ids,
            cache,
            checkpoint["clinical_scaling"],
            checkpoint["channel_scaling"],
        )
        model_config = checkpoint["config"]
        model = EEGClinicalNet(
            clinical_features=len(model_config["clinical_features"]),
            channels=int(model_config["channels"]),
            hidden_size=int(model_config["hidden_size"]),
            dropout=float(model_config["dropout"]),
            mode=mode,
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        labels, probabilities = _participant_probabilities(
            model, subjects, device, int(model_config["batch_size"]) * 2
        )
        if y_true is None:
            y_true = labels
        elif not np.array_equal(y_true, labels):
            raise RuntimeError("Baseline models were evaluated on different participants.")
        results[mode] = participant_metrics(labels, probabilities)
        prediction_records[mode] = {
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
            "predictions": probabilities.argmax(axis=1).tolist(),
        }

    assert y_true is not None
    test_frame = frame.set_index("participant_id").loc[test_ids]
    split_summary = {
        "seed": checkpoints["multimodal"]["config"]["seed"],
        "participant_counts": {
            partition: len(ids)
            for partition, ids in reference_split.items()
        },
        "class_counts": {
            partition: {
                label: int(
                    frame[
                        frame["participant_id"].isin(ids)
                        & frame["diagnosis"].eq(label)
                    ].shape[0]
                )
                for label in CLASS_NAMES
            }
            for partition, ids in reference_split.items()
        },
        "test_class_counts": {
            label: int(test_frame["diagnosis"].eq(label).sum())
            for label in CLASS_NAMES
        },
        "window_predictions_aggregated_to_participant": True,
        "participant_ids_disjoint_across_splits": True,
    }
    summary = {
        "dataset": {
            "dataset_id": raw_inventory["dataset_id"],
            "source": raw_inventory["source"],
            "license": raw_inventory["license"],
            "participant_count": raw_inventory["participants"],
            "class_counts": raw_inventory["class_counts"],
        },
        "split": split_summary,
        "evaluation": results,
        "model_predictions": prediction_records,
        "scope_and_limitations": [
            "This is cross-sectional AD/FTD/CN classification, not MCI-to-AD progression prediction.",
            "The held-out test contains participants from the same 88-person source cohort; no external validation was performed.",
            "MMSE is strongly associated with the source diagnostic labels. The clinical-only comparator and EEG-only comparator are reported to expose this shortcut risk.",
            "Confidence intervals are participant-level stratified bootstrap intervals and are wide/unstable at this cohort size.",
            "Attention weights and input gradients are model diagnostics, not validated causal explanations.",
            "Outputs are research classifications, not diagnoses or calibrated clinical risks.",
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "model_predictions": "saved in metrics.json"}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/openneuro_ds004504")
    parser.add_argument("--cache", default="data/processed/ds004504")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--output", default="evaluation/metrics.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    results = evaluate_checkpoints(
        args.data,
        args.cache,
        args.checkpoints,
        args.output,
        device_name=args.device,
    )
    from evaluation.eeg_plots import generate_evaluation_figures

    generate_evaluation_figures(results)


if __name__ == "__main__":
    main()
