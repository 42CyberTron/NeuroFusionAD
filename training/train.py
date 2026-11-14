"""Train using an explicitly supplied, authorized patient-visit manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.manifest import load_manifest, write_inventory
from models.network import NeuroFusionNet
from training.data import (
    PatientSequenceDataset,
    collate_sequences,
    fit_clinical_scaling,
    patient_splits,
)


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    diagnosis_loss: nn.Module,
) -> torch.Tensor:
    classification = diagnosis_loss(output["diagnosis_logits"], batch["diagnosis"])
    targets = batch["progression"]
    valid = torch.isfinite(targets)
    if valid.any():
        progression = nn.functional.binary_cross_entropy_with_logits(
            output["progression_logits"][valid], targets[valid]
        )
    else:
        progression = classification.new_zeros(())
    return classification + progression


def train(manifest_path: str, config_path: str, checkpoint_path: str) -> Path:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    frame = load_manifest(manifest_path)
    manifest_hash = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    write_inventory(frame, Path(checkpoint_path).parent / "dataset_inventory.json")
    horizons = [int(value) for value in config["horizons_months"]]
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("Prediction horizons must be positive month counts.")
    if len(set(horizons)) != len(horizons):
        raise ValueError("Prediction horizons must not contain duplicates.")
    if len(config["diagnosis_labels"]) != len(set(config["diagnosis_labels"])):
        raise ValueError("Diagnosis labels must be unique.")
    unsupported_labels = set(frame["diagnosis"]) - set(config["diagnosis_labels"])
    if unsupported_labels:
        raise ValueError(
            "Training configuration is missing diagnosis labels: "
            + ", ".join(sorted(unsupported_labels))
        )
    for horizon in horizons:
        column = f"conversion_{horizon}m"
        if column not in frame.columns:
            raise ValueError(f"Manifest requires a {column} column with 0/1 or blank labels.")
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not values.isin([0, 1]).all():
            raise ValueError(f"{column} labels must be 0, 1, or blank.")

    train_ids, val_ids, test_ids = patient_splits(frame, seed)
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Patient leakage detected between data splits.")
    features = list(config["clinical_features"])
    for horizon in horizons:
        column = f"conversion_{horizon}m"
        train_labels = pd.to_numeric(
            frame.loc[
                frame["patient_id"].astype(str).isin(train_ids)
                & frame["diagnosis"].eq("MCI"),
                column,
            ],
            errors="coerce",
        )
        if not train_labels.isin([0, 1]).any():
            raise ValueError(
                f"Training split has no observed {column} labels; cannot train this horizon."
            )
    scaling = fit_clinical_scaling(frame, train_ids, features)
    shared = {
        "features": features,
        "scaling": scaling,
        "horizons": horizons,
        "diagnosis_labels": list(config["diagnosis_labels"]),
        "image_size": int(config["image_size"]),
        "max_visits": int(config["max_visits"]),
    }
    datasets = {
        name: PatientSequenceDataset(frame, ids, **shared)
        for name, ids in (("train", train_ids), ("validation", val_ids), ("test", test_ids))
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=int(config["batch_size"]), shuffle=True,
            collate_fn=collate_sequences,
        ),
        "validation": DataLoader(
            datasets["validation"], batch_size=int(config["batch_size"]), shuffle=False,
            collate_fn=collate_sequences,
        ),
    }
    model = NeuroFusionNet(
        clinical_features=len(features),
        hidden_size=int(config["hidden_size"]),
        num_diagnoses=len(config["diagnosis_labels"]),
        num_horizons=len(horizons),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    candidate_checkpoint = checkpoint.with_name(checkpoint.name + ".partial")

    for epoch in range(int(config["epochs"])):
        model.train()
        train_loss = 0.0
        for raw_batch in loaders["train"]:
            batch = _move(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**{key: batch[key] for key in (
                "clinical", "mri", "pet", "modality_mask", "intervals_days", "visit_mask"
            )})
            loss = _loss(output, batch, criterion)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += float(loss.detach())

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for raw_batch in loaders["validation"]:
                batch = _move(raw_batch, device)
                output = model(**{key: batch[key] for key in (
                    "clinical", "mri", "pet", "modality_mask", "intervals_days", "visit_mask"
                )})
                validation_loss += float(_loss(output, batch, criterion))
        validation_loss /= max(len(loaders["validation"]), 1)
        print(
            f"epoch={epoch + 1}/{config['epochs']} "
            f"train_loss={train_loss / max(len(loaders['train']), 1):.4f} "
            f"validation_loss={validation_loss:.4f}"
        )
        if np.isfinite(validation_loss) and validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": config,
                    "clinical_scaling": scaling,
                    "dataset_manifest": str(Path(manifest_path).resolve()),
                    "dataset_manifest_sha256": manifest_hash,
                    "dataset_patient_counts": {
                        "train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)
                    },
                    "split_patient_ids": {
                        "train": sorted(train_ids),
                        "validation": sorted(val_ids),
                        "test": sorted(test_ids),
                    },
                    "best_validation_loss": best_loss,
                    "trained": True,
                },
                candidate_checkpoint,
            )
    if not candidate_checkpoint.is_file():
        raise RuntimeError(
            "Training produced no finite validation checkpoint; no weights were saved."
        )
    candidate_checkpoint.replace(checkpoint)
    print(f"Saved trained checkpoint: {checkpoint.resolve()}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Authorized patient-visit CSV")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--checkpoint", default="checkpoints/neurofusion.pt")
    args = parser.parse_args()
    train(args.manifest, args.config, args.checkpoint)


if __name__ == "__main__":
    main()
