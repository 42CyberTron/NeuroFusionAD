"""Train participant-split EEG-only, clinical-only, and attention-fusion models."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets.openneuro import CLASS_NAMES, load_dataset_index
from models.eeg_network import EEGClinicalNet
from training.eeg_data import (
    EEGSubject,
    EEGWindowDataset,
    create_participant_split,
    fit_channel_scaling,
    fit_clinical_scaling,
    load_subjects,
    participant_balanced_weights,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "eeg.json"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _participant_probabilities(
    model: EEGClinicalNet,
    subjects: list[EEGSubject],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[int] = []
    with torch.no_grad():
        for subject in subjects:
            window_probabilities: list[np.ndarray] = []
            for start in range(0, len(subject.windows), batch_size):
                windows = torch.from_numpy(
                    subject.windows[start : start + batch_size]
                ).to(device)
                clinical = torch.from_numpy(
                    np.repeat(
                        subject.clinical[None, :],
                        len(windows),
                        axis=0,
                    )
                ).to(device)
                mask = torch.tensor(
                    [[True, subject.clinical_present]] * len(windows),
                    dtype=torch.bool,
                    device=device,
                )
                output = model(windows, clinical, mask)
                window_probabilities.append(
                    torch.softmax(output["logits"], dim=1).cpu().numpy()
                )
            probabilities.append(
                np.concatenate(window_probabilities, axis=0).mean(axis=0)
            )
            labels.append(subject.label)
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


def _train_mode(
    mode: str,
    train_subjects: list[EEGSubject],
    validation_subjects: list[EEGSubject],
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    common_checkpoint: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    _seed_everything(seed)
    dataset = EEGWindowDataset(train_subjects)
    sampler = WeightedRandomSampler(
        participant_balanced_weights(dataset, train_subjects),
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(int(config["batch_size"]), len(dataset)),
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) > int(config["batch_size"]),
    )
    model = EEGClinicalNet(
        clinical_features=len(config["clinical_features"]),
        channels=int(config["channels"]),
        hidden_size=int(config["hidden_size"]),
        dropout=float(config["dropout"]),
        mode=mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    patience = int(config["patience"])
    batch_size = int(config["batch_size"])

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        batch_losses: list[float] = []
        for eeg, clinical, labels, modality_mask in loader:
            eeg = eeg.to(device, non_blocking=True)
            clinical = clinical.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            modality_mask = modality_mask.to(device, non_blocking=True)
            if mode == "multimodal":
                # Mask true clinical modality availability during training; the
                # EEG and metadata remain observations from real participants.
                dropped = torch.rand(len(modality_mask), device=device) < 0.15
                modality_mask[dropped, 1] = False
            optimizer.zero_grad(set_to_none=True)
            logits = model(eeg, clinical, modality_mask)["logits"]
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        validation_labels, validation_probabilities = _participant_probabilities(
            model, validation_subjects, device, batch_size * 2
        )
        validation_loss = float(
            -np.log(
                np.clip(
                    validation_probabilities[
                        np.arange(len(validation_labels)), validation_labels
                    ],
                    1e-8,
                    1.0,
                )
            ).mean()
        )
        scheduler.step(validation_loss)
        improved = validation_loss < best_loss - 1e-4
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 5 == 0 or improved:
            print(
                f"{mode:12s} epoch {epoch:02d}/{config['epochs']} "
                f"window_loss={np.mean(batch_losses):.4f} "
                f"validation_participant_logloss={validation_loss:.4f} "
                f"best={best_loss:.4f}",
                flush=True,
            )
        if stale_epochs >= patience:
            print(f"{mode}: early stopping at epoch {epoch}.", flush=True)
            break
    if best_state is None:
        raise RuntimeError(f"{mode} training did not produce a validation checkpoint.")

    model.load_state_dict(best_state, strict=True)
    model.eval()
    checkpoint = {
        **common_checkpoint,
        "trained": True,
        "mode": mode,
        "state_dict": best_state,
        "training": {
            "best_epoch": best_epoch,
            "best_validation_participant_logloss": best_loss,
            "epochs_run": epoch,
            "early_stopping_patience": patience,
            "train_participant_count": len(train_subjects),
            "validation_participant_count": len(validation_subjects),
            "train_window_count": len(dataset),
            "participant_balanced_sampling": True,
            "clinical_modality_dropout_probability": 0.15 if mode == "multimodal" else 0.0,
        },
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint["training"]


def train_models(
    data_root: str | Path = "data/openneuro_ds004504",
    cache_dir: str | Path = "data/processed/ds004504",
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_dir: str | Path = "checkpoints",
    *,
    device_name: str | None = None,
) -> dict[str, Any]:
    root = Path(data_root)
    cache = Path(cache_dir)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    frame, inventory = load_dataset_index(root)
    missing_caches = [
        participant_id
        for participant_id in frame["participant_id"]
        if not (cache / f"{participant_id}.npz").is_file()
    ]
    if missing_caches:
        raise FileNotFoundError(
            f"{len(missing_caches)} preprocessed participant caches are missing. "
            "Run `python -m preprocessing.process_openneuro` first."
        )
    processed_inventory_path = cache / "dataset_inventory.json"
    if not processed_inventory_path.is_file():
        raise FileNotFoundError(
            f"Processed inventory is missing at {processed_inventory_path}. "
            "Run `python -m preprocessing.process_openneuro` first."
        )
    processed_inventory = json.loads(
        processed_inventory_path.read_text(encoding="utf-8")
    )

    seed = int(config["seed"])
    split = create_participant_split(
        frame,
        seed=seed,
        test_fraction=float(config["test_fraction"]),
        validation_fraction_of_remainder=float(
            config["validation_fraction_of_remainder"]
        ),
    )
    clinical_scaling = fit_clinical_scaling(frame, split["train"])
    channel_scaling = fit_channel_scaling(cache, split["train"])
    train_subjects = load_subjects(
        frame, split["train"], cache, clinical_scaling, channel_scaling
    )
    validation_subjects = load_subjects(
        frame, split["validation"], cache, clinical_scaling, channel_scaling
    )
    if set(split["train"]) & set(split["validation"]) or set(split["train"]) & set(split["test"]):
        raise RuntimeError("Training and held-out participants overlap.")

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    metadata_bytes = (
        (root / "participants.tsv").read_bytes()
        + (root / "dataset_description.json").read_bytes()
    )
    common_checkpoint = {
        "dataset_id": "OpenNeuro ds004504 / NEMAR on004504",
        "dataset_license": inventory["license"],
        "dataset_inventory": processed_inventory,
        "cohort_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "config": config,
        "clinical_scaling": clinical_scaling,
        "channel_scaling": channel_scaling,
        "split_patient_ids": split,
        "class_names": list(CLASS_NAMES),
        "preprocessing": processed_inventory.get("preprocessing", {}),
        "device_used_for_training": str(device),
    }
    checkpoint_root = Path(checkpoint_dir)
    training_results: dict[str, Any] = {}
    checkpoint_names = {
        "multimodal": "eeg_multimodal.pt",
        "eeg_only": "eeg_only.pt",
        "clinical_only": "clinical_only.pt",
    }
    for offset, mode in enumerate(("multimodal", "eeg_only", "clinical_only")):
        output_path = checkpoint_root / checkpoint_names[mode]
        training_results[mode] = _train_mode(
            mode,
            train_subjects,
            validation_subjects,
            config=config,
            checkpoint_path=output_path,
            common_checkpoint=common_checkpoint,
            device=device,
            seed=seed + offset,
        )
        print(f"Saved trained {mode} weights to {output_path}.", flush=True)

    split_path = Path("evaluation") / "participant_splits.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "counts": {name: len(ids) for name, ids in split.items()},
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
                    for partition, ids in split.items()
                },
                "participant_ids": split,
                "test_set_used_for_model_selection": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"device": str(device), "training": training_results}, indent=2))
    return training_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/openneuro_ds004504")
    parser.add_argument("--cache", default="data/processed/ds004504")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available, else CPU.")
    args = parser.parse_args()
    train_models(
        args.data,
        args.cache,
        args.config,
        args.checkpoints,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
