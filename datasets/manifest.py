"""Validate and summarize a locally authorized research dataset manifest."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.clinical import (
    CLINICAL_RANGES,
    SEX_FEATURES,
    encode_sex,
    validate_numeric_clinical,
)


IDENTIFIERS = ("patient_id", "visit_date", "diagnosis")
IMAGE_COLUMNS = ("mri_path", "pet_path")
CLINICAL_COLUMNS = (
    "age",
    *SEX_FEATURES,
    "education_years",
    "mmse",
    "adas13",
    "cdrsb",
    "faq",
    "apoe4",
    "hippocampal_volume",
    "brain_volume",
)


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    frame = pd.read_csv(manifest_path)
    missing = [column for column in IDENTIFIERS if column not in frame.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Dataset manifest contains no visits.")
    if frame["patient_id"].isna().any() or frame["visit_date"].isna().any():
        raise ValueError("patient_id and visit_date must be present for every visit.")
    frame["patient_id"] = frame["patient_id"].astype(str).str.strip()
    if frame["patient_id"].eq("").any():
        raise ValueError("patient_id cannot be blank.")
    frame["visit_date"] = pd.to_datetime(frame["visit_date"], errors="raise")
    frame["diagnosis"] = frame["diagnosis"].astype(str).str.upper().str.strip()
    invalid = sorted(set(frame["diagnosis"]) - {"CN", "MCI", "AD"})
    if invalid:
        raise ValueError(f"Unsupported diagnosis labels: {', '.join(invalid)}")
    if frame.duplicated(["patient_id", "visit_date"]).any():
        raise ValueError("A patient cannot have more than one visit on the same date.")
    if "sex" in frame.columns:
        if any(feature in frame.columns for feature in SEX_FEATURES):
            raise ValueError("Provide sex categories or one-hot columns, not both.")
        encodings = [
            encode_sex(None if pd.isna(value) else value)
            for value in frame["sex"]
        ]
        for feature in SEX_FEATURES:
            frame[feature] = [encoding[feature] for encoding in encodings]
        frame = frame.drop(columns=["sex"])
    else:
        for feature in SEX_FEATURES:
            if feature not in frame.columns:
                frame[feature] = pd.NA
    for column in CLINICAL_COLUMNS:
        if column in frame.columns:
            converted = pd.to_numeric(frame[column], errors="coerce")
            invalid_values = frame[column].notna() & converted.isna()
            non_finite = converted.notna() & ~np.isfinite(converted)
            if invalid_values.any() or non_finite.any():
                raise ValueError(f"{column} contains non-numeric clinical values.")
            frame[column] = converted
    for column, (minimum, maximum) in CLINICAL_RANGES.items():
        if column not in frame.columns:
            continue
        values = frame[column].dropna()
        if minimum is not None and (values < minimum).any():
            raise ValueError(f"{column} contains values below {minimum:g}.")
        if maximum is not None and (values > maximum).any():
            raise ValueError(f"{column} contains values above {maximum:g}.")
        if column == "apoe4" and (values % 1 != 0).any():
            raise ValueError("apoe4 must contain integer allele counts (0, 1, or 2).")
        if column in ("hippocampal_volume", "brain_volume") and (values <= 0).any():
            raise ValueError(f"{column} values must be positive or blank.")
    for _, row in frame.iterrows():
        validate_numeric_clinical(row.to_dict())
    for column in IMAGE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    for column in IMAGE_COLUMNS:
        populated = frame[column].dropna().astype(str)
        missing_files = [item for item in populated if not Path(item).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"{column} refers to missing files; first missing path: {missing_files[0]}"
            )
    for column in frame.columns:
        if column.startswith("conversion_"):
            converted = pd.to_numeric(frame[column], errors="coerce")
            invalid_values = frame[column].notna() & converted.isna()
            non_finite = converted.notna() & ~np.isfinite(converted)
            if (
                invalid_values.any()
                or non_finite.any()
                or not converted.dropna().isin([0, 1]).all()
            ):
                raise ValueError(f"{column} must contain only 0, 1, or blank.")
            frame[column] = converted
    return frame


def build_inventory(frame: pd.DataFrame) -> dict[str, Any]:
    clinical = [
        column
        for column in frame.columns
        if column not in (*IDENTIFIERS, *IMAGE_COLUMNS)
        and not column.startswith("conversion_")
    ]
    return {
        "source": "User-provided local manifest; dataset identity is not inferred.",
        "patient_count": int(frame["patient_id"].nunique()),
        "visit_count": int(len(frame)),
        "modalities": {
            "mri": {
                "visits_present": int(frame["mri_path"].notna().sum()),
                "visits_missing": int(frame["mri_path"].isna().sum()),
            },
            "pet": {
                "visits_present": int(frame["pet_path"].notna().sum()),
                "visits_missing": int(frame["pet_path"].isna().sum()),
            },
        },
        "diagnosis_visit_counts": {
            str(label): int(count)
            for label, count in frame["diagnosis"].value_counts(dropna=False).items()
        },
        "clinical_columns": clinical,
        "clinical_missing_values": {
            column: int(frame[column].isna().sum()) for column in clinical
        },
        "conversion_label_counts": {
            column: {
                str(label): int(count)
                for label, count in frame[column].value_counts(dropna=False).items()
            }
            for column in frame.columns
            if column.startswith("conversion_")
        },
    }


def write_inventory(frame: pd.DataFrame, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build_inventory(frame), indent=2), encoding="utf-8"
    )
    return destination
