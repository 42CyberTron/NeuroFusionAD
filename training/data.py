"""Patient-level sequence dataset and cohort splits."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from preprocessing.imaging import load_nifti_slice


def patient_splits(
    frame: pd.DataFrame, seed: int
) -> tuple[set[str], set[str], set[str]]:
    patients = frame[["patient_id"]].drop_duplicates().reset_index(drop=True)
    groups = patients["patient_id"].astype(str).to_numpy()
    first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(first.split(patients, groups=groups))
    remaining = patients.iloc[train_idx].reset_index(drop=True)
    remaining_groups = remaining["patient_id"].astype(str).to_numpy()
    second = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + 1)
    train_rel, val_rel = next(second.split(remaining, groups=remaining_groups))
    return (
        set(remaining.iloc[train_rel]["patient_id"].astype(str)),
        set(remaining.iloc[val_rel]["patient_id"].astype(str)),
        set(patients.iloc[test_idx]["patient_id"].astype(str)),
    )


def fit_clinical_scaling(
    frame: pd.DataFrame, patient_ids: set[str], features: list[str]
) -> dict[str, dict[str, float]]:
    train_rows = frame[frame["patient_id"].astype(str).isin(patient_ids)]
    scaling: dict[str, dict[str, float]] = {}
    for feature in features:
        if feature not in frame.columns:
            raise ValueError(f"Manifest is missing clinical feature column: {feature}")
        values = pd.to_numeric(train_rows[feature], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        filled = values.fillna(median)
        scale = float(filled.std(ddof=0)) if len(filled) else 1.0
        scaling[feature] = {"median": median, "scale": scale if scale > 1e-8 else 1.0}
    return scaling


class PatientSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        patient_ids: set[str],
        features: list[str],
        scaling: dict[str, dict[str, float]],
        horizons: list[int],
        diagnosis_labels: list[str],
        image_size: int,
        max_visits: int,
    ) -> None:
        self.frame = frame[frame["patient_id"].astype(str).isin(patient_ids)].copy()
        self.features = features
        self.scaling = scaling
        self.horizons = horizons
        self.diagnosis_labels = diagnosis_labels
        self.image_size = image_size
        self.max_visits = max_visits
        self.patient_groups = []
        for _, full_history in self.frame.groupby("patient_id", sort=True):
            full_history = full_history.sort_values("visit_date").reset_index(drop=True)
            self.patient_groups.extend(
                full_history.iloc[: anchor + 1]
                for anchor in range(len(full_history))
            )
        if not self.patient_groups:
            raise ValueError("No patients are available for this dataset split.")

    def __len__(self) -> int:
        return len(self.patient_groups)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        visits = self.patient_groups[index].tail(self.max_visits).reset_index(drop=True)
        count = len(visits)
        size = self.image_size
        clinical = np.zeros((count, len(self.features)), dtype=np.float32)
        mri = torch.zeros((count, 1, size, size), dtype=torch.float32)
        pet = torch.zeros_like(mri)
        mask = torch.zeros((count, 3), dtype=torch.bool)
        elapsed = np.zeros(count, dtype=np.float32)
        start = visits.iloc[0]["visit_date"]
        for idx, row in visits.iterrows():
            observed = 0
            for feature_idx, feature in enumerate(self.features):
                raw = pd.to_numeric(pd.Series([row.get(feature)]), errors="coerce").iloc[0]
                if pd.notna(raw):
                    observed += 1
                stats = self.scaling[feature]
                value = stats["median"] if pd.isna(raw) else float(raw)
                clinical[idx, feature_idx] = (value - stats["median"]) / stats["scale"]
            mask[idx, 0] = observed > 0
            for column, modality_idx, target in (
                ("mri_path", 1, mri),
                ("pet_path", 2, pet),
            ):
                image_path = row.get(column)
                if pd.notna(image_path) and str(image_path).strip():
                    target[idx], _ = load_nifti_slice(str(image_path), size)
                    mask[idx, modality_idx] = True
            elapsed[idx] = (row["visit_date"] - start).total_seconds() / 86400.0
        if not mask.any():
            raise ValueError(
                f"Patient {visits.iloc[-1]['patient_id']} has no observed clinical or imaging input."
            )

        latest = visits.iloc[-1]
        diagnosis_name = str(latest["diagnosis"]).upper()
        diagnosis = self.diagnosis_labels.index(diagnosis_name)
        targets = []
        for horizon in self.horizons:
            column = f"conversion_{horizon}m"
            if column not in visits.columns:
                raise ValueError(f"Manifest requires a {column} outcome column.")
            targets.append(
                float(latest[column])
                if diagnosis_name == "MCI" and pd.notna(latest[column])
                else float("nan")
            )
        return {
            "clinical": torch.from_numpy(clinical),
            "mri": mri,
            "pet": pet,
            "modality_mask": mask,
            "intervals_days": torch.from_numpy(elapsed),
            "visit_mask": torch.ones(count, dtype=torch.bool),
            "diagnosis": torch.tensor(diagnosis, dtype=torch.long),
            "progression": torch.tensor(targets, dtype=torch.float32),
        }


def collate_sequences(
    samples: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    max_length = max(item["clinical"].shape[0] for item in samples)
    result: dict[str, torch.Tensor] = {}
    for key in (
        "clinical",
        "mri",
        "pet",
        "modality_mask",
        "intervals_days",
        "visit_mask",
    ):
        exemplar = samples[0][key]
        shape = (len(samples), max_length, *exemplar.shape[1:])
        batch = torch.zeros(shape, dtype=exemplar.dtype)
        for idx, item in enumerate(samples):
            batch[idx, : item[key].shape[0]] = item[key]
        result[key] = batch
    result["diagnosis"] = torch.stack([item["diagnosis"] for item in samples])
    result["progression"] = torch.stack([item["progression"] for item in samples])
    return result
