"""Evaluate a trained checkpoint on its patient-level held-out test split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.manifest import load_manifest
from evaluation.metrics import binary_metrics, multiclass_metrics
from models.network import NeuroFusionNet
from training.data import PatientSequenceDataset, collate_sequences


def evaluate(checkpoint_path: str, manifest_path: str, output_path: str) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not checkpoint.get("trained"):
        raise ValueError("Cannot evaluate a checkpoint not marked as trained.")
    config = checkpoint["config"]
    manifest_hash = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    if manifest_hash != checkpoint.get("dataset_manifest_sha256"):
        raise ValueError(
            "Evaluation manifest differs from the exact manifest used for training."
        )
    frame = load_manifest(manifest_path)
    stored_splits = checkpoint.get("split_patient_ids")
    if not stored_splits or "test" not in stored_splits:
        raise ValueError("Checkpoint does not contain its patient-level held-out test split.")
    test_ids = set(stored_splits["test"])
    available_patients = set(frame["patient_id"].astype(str))
    if not test_ids.issubset(available_patients):
        raise ValueError("Manifest is missing patient IDs from the checkpoint test split.")
    dataset = PatientSequenceDataset(
        frame,
        test_ids,
        list(config["clinical_features"]),
        checkpoint["clinical_scaling"],
        [int(value) for value in config["horizons_months"]],
        list(config["diagnosis_labels"]),
        int(config["image_size"]),
        int(config["max_visits"]),
    )
    loader = DataLoader(
        dataset, batch_size=int(config["batch_size"]), shuffle=False,
        collate_fn=collate_sequences,
    )
    model = NeuroFusionNet(
        clinical_features=len(config["clinical_features"]),
        hidden_size=int(config["hidden_size"]),
        num_diagnoses=len(config["diagnosis_labels"]),
        num_horizons=len(config["horizons_months"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    diagnoses: list[np.ndarray] = []
    diagnosis_probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    progression_probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(**{key: batch[key] for key in (
                "clinical", "mri", "pet", "modality_mask", "intervals_days", "visit_mask"
            )})
            diagnoses.extend(batch["diagnosis"].numpy())
            diagnosis_probs.extend(torch.softmax(output["diagnosis_logits"], -1).numpy())
            labels.extend(batch["progression"].numpy())
            progression_probs.extend(torch.sigmoid(output["progression_logits"]).numpy())
    diagnosis_array = np.asarray(diagnoses)
    diagnosis_probability_array = np.asarray(diagnosis_probs)
    label_array = np.asarray(labels)
    progression_probability_array = np.asarray(progression_probs)
    results = {
        "dataset_manifest": str(Path(manifest_path).resolve()),
        "test_patient_count": len(test_ids),
        "diagnosis": multiclass_metrics(diagnosis_array, diagnosis_probability_array),
        "progression_by_horizon": {
            f"{int(horizon)}_month": binary_metrics(
                label_array[:, index], progression_probability_array[:, index]
            )
            for index, horizon in enumerate(config["horizons_months"])
        },
        "calibration_note": "Brier scores are reported; no calibration transform is applied.",
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/neurofusion.pt")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="evaluation/metrics.json")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.manifest, args.output)


if __name__ == "__main__":
    main()
