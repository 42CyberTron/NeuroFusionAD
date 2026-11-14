"""Participant-level split, train-only normalization, and window datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from datasets.openneuro import CLASS_NAMES


@dataclass
class EEGSubject:
    participant_id: str
    label: int
    clinical: np.ndarray
    clinical_present: bool
    windows: np.ndarray


def create_participant_split(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    test_fraction: float = 0.2,
    validation_fraction_of_remainder: float = 0.2,
) -> dict[str, list[str]]:
    """Split unique participants before windows are exposed to model training."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if not 0 < validation_fraction_of_remainder < 1:
        raise ValueError("validation_fraction_of_remainder must be between 0 and 1.")
    if frame["participant_id"].duplicated().any():
        raise ValueError("Participant split requires one metadata row per participant.")
    ids = frame["participant_id"].astype(str).to_numpy()
    labels = frame["diagnosis"].astype(str).to_numpy()
    if len(set(labels)) != len(CLASS_NAMES) or set(labels) != set(CLASS_NAMES):
        raise ValueError(f"Split requires exactly the classes {CLASS_NAMES}.")

    train_and_validation, test_ids = train_test_split(
        ids,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,
    )
    train_labels = (
        frame.set_index("participant_id").loc[train_and_validation, "diagnosis"].to_numpy()
    )
    train_ids, validation_ids = train_test_split(
        train_and_validation,
        test_size=validation_fraction_of_remainder,
        random_state=seed + 1,
        stratify=train_labels,
    )
    split = {
        "train": sorted(str(value) for value in train_ids),
        "validation": sorted(str(value) for value in validation_ids),
        "test": sorted(str(value) for value in test_ids),
    }
    all_ids = [value for partition in split.values() for value in partition]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(ids):
        raise RuntimeError("Participant split is overlapping or incomplete.")
    return split


def fit_clinical_scaling(
    frame: pd.DataFrame, train_ids: Sequence[str]
) -> dict[str, dict[str, float]]:
    train = frame[frame["participant_id"].astype(str).isin(set(train_ids))]
    result: dict[str, dict[str, float]] = {}
    for column in ("age", "mmse"):
        values = pd.to_numeric(train[column], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Training participants contain no valid {column} values.")
        scale = float(values.std())
        result[column] = {
            "mean": float(values.mean()),
            "scale": scale if scale > 1e-8 else 1.0,
        }
    return result


def encode_clinical(
    values: dict[str, Any] | pd.Series,
    scaling: dict[str, dict[str, float]],
) -> tuple[np.ndarray, bool]:
    """Encode age, MMSE, sex, and explicit missingness indicators."""
    record = dict(values)

    def optional_number(name: str, minimum: float, maximum: float) -> float | None:
        raw = record.get(name)
        if raw is None or pd.isna(raw) or (isinstance(raw, str) and not raw.strip()):
            return None
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric or missing.") from exc
        if not np.isfinite(number) or not minimum <= number <= maximum:
            raise ValueError(f"{name} must be finite and between {minimum:g} and {maximum:g}.")
        return number

    age = optional_number("age", 18, 120)
    mmse = optional_number("mmse", 0, 30)
    gender_raw = record.get("gender")
    gender = "" if gender_raw is None or pd.isna(gender_raw) else str(gender_raw).strip().upper()
    if gender in {"", "NOT RECORDED", "OTHER"}:
        gender = ""
    elif gender not in {"M", "F", "MALE", "FEMALE"}:
        raise ValueError("gender must be Male, Female, Other, or Not recorded.")
    else:
        gender = "M" if gender in {"M", "MALE"} else "F"

    clinical = np.asarray(
        [
            0.0 if age is None else (age - scaling["age"]["mean"]) / scaling["age"]["scale"],
            0.0 if mmse is None else (mmse - scaling["mmse"]["mean"]) / scaling["mmse"]["scale"],
            float(gender == "M"),
            float(gender == "F"),
            float(age is None),
            float(mmse is None),
            float(gender == ""),
        ],
        dtype=np.float32,
    )
    if not np.isfinite(clinical).all():
        raise ValueError("Clinical encoding produced non-finite features.")
    return clinical, bool(age is not None or mmse is not None or gender)


def fit_channel_scaling(
    cache_dir: str | Path, train_ids: Sequence[str]
) -> dict[str, list[float]]:
    """Estimate channel-wise mean and standard deviation from training people only."""
    channel_sum: np.ndarray | None = None
    channel_sum_squares: np.ndarray | None = None
    sample_count = 0
    for participant_id in train_ids:
        cache_path = Path(cache_dir) / f"{participant_id}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Preprocessed cache is missing: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as data:
            windows = np.asarray(data["windows_uv"], dtype=np.float64)
        if windows.ndim != 3 or not len(windows) or not np.isfinite(windows).all():
            raise ValueError(f"Invalid preprocessed windows for {participant_id}.")
        current_sum = windows.sum(axis=(0, 2), dtype=np.float64)
        current_sum_squares = np.square(windows).sum(axis=(0, 2), dtype=np.float64)
        channel_sum = current_sum if channel_sum is None else channel_sum + current_sum
        channel_sum_squares = (
            current_sum_squares
            if channel_sum_squares is None
            else channel_sum_squares + current_sum_squares
        )
        sample_count += int(windows.shape[0] * windows.shape[2])
    if channel_sum is None or channel_sum_squares is None or sample_count == 0:
        raise ValueError("Cannot fit EEG channel scaling without training windows.")
    mean = channel_sum / sample_count
    variance = np.maximum(channel_sum_squares / sample_count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    if not np.isfinite(scale).all() or np.any(scale < 1e-6):
        raise ValueError("At least one EEG channel is flat in the training set.")
    return {"mean_uv": mean.tolist(), "scale_uv": scale.tolist()}


def standardize_windows(
    windows: np.ndarray, scaling: dict[str, list[float]]
) -> np.ndarray:
    mean = np.asarray(scaling["mean_uv"], dtype=np.float32)[None, :, None]
    scale = np.asarray(scaling["scale_uv"], dtype=np.float32)[None, :, None]
    if windows.shape[1] != len(mean[0]):
        raise ValueError("EEG channel count does not match the trained model.")
    standardized = (windows.astype(np.float32, copy=False) - mean) / scale
    if not np.isfinite(standardized).all():
        raise ValueError("EEG normalization produced non-finite values.")
    return standardized


def load_subjects(
    frame: pd.DataFrame,
    participant_ids: Sequence[str],
    cache_dir: str | Path,
    clinical_scaling: dict[str, dict[str, float]],
    channel_scaling: dict[str, list[float]],
) -> list[EEGSubject]:
    by_id = frame.set_index("participant_id", drop=False)
    subjects: list[EEGSubject] = []
    for participant_id in participant_ids:
        if participant_id not in by_id.index:
            raise ValueError(f"Unknown participant in split: {participant_id}")
        row = by_id.loc[participant_id]
        with np.load(Path(cache_dir) / f"{participant_id}.npz", allow_pickle=False) as data:
            windows_uv = np.asarray(data["windows_uv"], dtype=np.float32)
        windows = standardize_windows(windows_uv, channel_scaling)
        clinical, clinical_present = encode_clinical(row, clinical_scaling)
        try:
            label = CLASS_NAMES.index(str(row["diagnosis"]))
        except ValueError as exc:
            raise ValueError(f"Unsupported diagnosis for {participant_id}.") from exc
        subjects.append(
            EEGSubject(
                participant_id=str(participant_id),
                label=label,
                clinical=clinical,
                clinical_present=clinical_present,
                windows=windows,
            )
        )
    return subjects


class EEGWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Flatten windows for optimization while retaining participant identities."""

    def __init__(self, subjects: Sequence[EEGSubject]) -> None:
        if not subjects:
            raise ValueError("At least one participant is required.")
        self.windows = np.concatenate([subject.windows for subject in subjects], axis=0)
        self.clinical = np.concatenate(
            [
                np.repeat(subject.clinical[None, :], len(subject.windows), axis=0)
                for subject in subjects
            ],
            axis=0,
        )
        self.labels = np.concatenate(
            [
                np.full(len(subject.windows), subject.label, dtype=np.int64)
                for subject in subjects
            ]
        )
        self.subject_indices = np.concatenate(
            [
                np.full(len(subject.windows), index, dtype=np.int64)
                for index, subject in enumerate(subjects)
            ]
        )
        self.clinical_present = np.concatenate(
            [
                np.full(len(subject.windows), subject.clinical_present, dtype=bool)
                for subject in subjects
            ]
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = torch.tensor([True, self.clinical_present[index]], dtype=torch.bool)
        return (
            torch.from_numpy(self.windows[index]),
            torch.from_numpy(self.clinical[index]),
            torch.tensor(self.labels[index], dtype=torch.long),
            mask,
        )


def participant_balanced_weights(
    dataset: EEGWindowDataset, subjects: Sequence[EEGSubject]
) -> torch.Tensor:
    class_subject_counts = np.bincount(
        [subject.label for subject in subjects], minlength=len(CLASS_NAMES)
    )
    window_counts = np.bincount(
        dataset.subject_indices, minlength=len(subjects)
    )
    weights = np.empty(len(dataset), dtype=np.float64)
    for index, subject_index in enumerate(dataset.subject_indices):
        label = subjects[int(subject_index)].label
        weights[index] = 1.0 / (
            class_subject_counts[label] * window_counts[int(subject_index)]
        )
    weights /= weights.sum()
    return torch.as_tensor(weights, dtype=torch.double)
