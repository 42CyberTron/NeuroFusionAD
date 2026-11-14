"""Validated metadata and BIDS index for the public ds004504 EEG cohort."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


DATASET_ID = "ds004504"
NEMAR_DATASET_ID = "on004504"
NEMAR_VERSION = "v1.0.0"
CLASS_NAMES = ("AD", "FTD", "CN")
GROUP_TO_CLASS = {"A": "AD", "F": "FTD", "C": "CN"}
CHANNELS = (
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
)


def load_dataset_index(root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read participants and verify every raw BIDS recording and sidecar."""
    data_root = Path(root)
    participants_path = data_root / "participants.tsv"
    description_path = data_root / "dataset_description.json"
    if not participants_path.is_file():
        raise FileNotFoundError(
            f"Dataset metadata is missing at {participants_path}. "
            "Run `python datasets/download_openneuro.py` first."
        )
    if not description_path.is_file():
        raise FileNotFoundError(f"Dataset description is missing: {description_path}")

    description = json.loads(description_path.read_text(encoding="utf-8"))
    license_name = str(description.get("License", "")).strip()
    if license_name.casefold() not in {"cc0", "cc0-1.0"}:
        raise ValueError(
            f"Expected the public NEMAR copy's CC0 license, received {license_name!r}."
        )

    participants = pd.read_csv(participants_path, sep="\t", dtype={"participant_id": str})
    required = {"participant_id", "Gender", "Age", "Group", "MMSE"}
    missing_columns = required - set(participants.columns)
    if missing_columns:
        raise ValueError(f"participants.tsv is missing columns: {sorted(missing_columns)}")
    if participants["participant_id"].isna().any():
        raise ValueError("Every participant must have an identifier.")
    participants["participant_id"] = participants["participant_id"].astype(str).str.strip()
    if participants["participant_id"].duplicated().any():
        raise ValueError("participants.tsv contains duplicate participant IDs.")
    if participants["participant_id"].isna().any() or participants["participant_id"].eq("").any():
        raise ValueError("Every participant must have an identifier.")

    participants["group_code"] = participants["Group"].astype(str).str.strip().str.upper()
    unknown_groups = sorted(set(participants["group_code"]) - set(GROUP_TO_CLASS))
    if unknown_groups:
        raise ValueError(f"Unsupported source group codes: {unknown_groups}")
    participants["diagnosis"] = participants["group_code"].map(GROUP_TO_CLASS)
    participants["gender"] = participants["Gender"].astype("string").str.strip().str.upper()
    participants["age"] = pd.to_numeric(participants["Age"], errors="coerce")
    participants["mmse"] = pd.to_numeric(participants["MMSE"], errors="coerce")
    for source, converted, name in (
        (participants["Age"], participants["age"], "Age"),
        (participants["MMSE"], participants["mmse"], "MMSE"),
    ):
        if (source.notna() & converted.isna()).any():
            raise ValueError(f"{name} contains a non-numeric non-missing value.")
    if not participants["age"].dropna().between(18, 120).all():
        raise ValueError("Age values must be between 18 and 120 years.")
    if not participants["mmse"].dropna().between(0, 30).all():
        raise ValueError("MMSE values must be between 0 and 30.")
    if not participants["gender"].dropna().isin(["M", "F"]).all():
        raise ValueError("Gender values must match the source codebook (M or F).")

    rows: list[dict[str, Any]] = []
    for participant in participants.to_dict(orient="records"):
        participant_id = str(participant["participant_id"])
        eeg_dir = data_root / participant_id / "eeg"
        eeg_files = sorted(eeg_dir.glob(f"{participant_id}_task-eyesclosed_eeg.set"))
        if len(eeg_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one raw eyes-closed .set file for {participant_id}; "
                f"found {len(eeg_files)}."
            )
        stem = eeg_files[0].name.removesuffix("_eeg.set")
        channels_path = eeg_dir / f"{stem}_channels.tsv"
        sidecar_path = eeg_dir / f"{stem}_eeg.json"
        for required_path in (channels_path, sidecar_path):
            if not required_path.is_file():
                raise FileNotFoundError(f"Required BIDS sidecar is missing: {required_path}")

        channel_table = pd.read_csv(channels_path, sep="\t")
        channel_names = channel_table["name"].astype(str).tolist()
        if "type" not in channel_table or not channel_table["type"].astype(str).eq("EEG").all():
            raise ValueError(f"{participant_id} has a non-EEG or incomplete channel table.")
        if "units" not in channel_table or not channel_table["units"].astype(str).str.casefold().eq("microv").all():
            raise ValueError(f"{participant_id} does not report microvolt EEG units.")
        if [name.casefold() for name in channel_names] != [
            name.casefold() for name in CHANNELS
        ]:
            raise ValueError(
                f"{participant_id} has an unexpected channel order or channel set."
            )
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if int(sidecar.get("EEGChannelCount", -1)) != len(CHANNELS):
            raise ValueError(f"{participant_id} sidecar does not report 19 EEG channels.")
        if float(sidecar.get("SamplingFrequency", 0)) != 500.0:
            raise ValueError(f"{participant_id} is not recorded at the expected 500 Hz.")
        if sidecar.get("RecordingType") != "continuous":
            raise ValueError(f"{participant_id} is not a continuous EEG recording.")
        if not pd.notna(sidecar.get("RecordingDuration")) or float(sidecar["RecordingDuration"]) <= 0:
            raise ValueError(f"{participant_id} has an invalid EEG recording duration.")

        rows.append(
            {
                "participant_id": participant_id,
                "diagnosis": participant["diagnosis"],
                "group_code": participant["group_code"],
                "gender": None if pd.isna(participant["gender"]) else str(participant["gender"]),
                "age": None if pd.isna(participant["age"]) else float(participant["age"]),
                "mmse": None if pd.isna(participant["mmse"]) else float(participant["mmse"]),
                "eeg_path": str(eeg_files[0].resolve()),
                "recording_duration_seconds": float(sidecar["RecordingDuration"]),
                "sampling_frequency_hz": float(sidecar["SamplingFrequency"]),
                "channel_count": int(sidecar["EEGChannelCount"]),
                "eeg_bytes": eeg_files[0].stat().st_size,
            }
        )

    frame = pd.DataFrame(rows)
    if len(frame) != 88:
        raise ValueError(f"Expected 88 participants in ds004504; found {len(frame)}.")
    observed_classes = set(frame["diagnosis"])
    if observed_classes != set(CLASS_NAMES):
        raise ValueError(f"Expected classes {CLASS_NAMES}; found {sorted(observed_classes)}.")

    source_datasets = description.get("SourceDatasets", [])
    source_doi = (
        str(source_datasets[0].get("DOI", ""))
        if source_datasets and isinstance(source_datasets[0], dict)
        else ""
    )
    openneuro_version = (
        source_doi.rsplit(".v", 1)[-1] if ".v" in source_doi else None
    )
    inventory = {
        "dataset_id": DATASET_ID,
        "source": "NEMAR public CC0 copy of OpenNeuro ds004504",
        "nemar_dataset_id": NEMAR_DATASET_ID,
        "nemar_path_version": NEMAR_VERSION,
        "nemar_dataset_version": description.get("Version"),
        "openneuro_dataset_version": openneuro_version,
        "bids_version": description.get("BIDSVersion"),
        "license": license_name,
        "participants": int(len(frame)),
        "raw_eeg_recordings": int(frame["eeg_path"].nunique()),
        "raw_eeg_bytes": int(frame["eeg_bytes"].sum()),
        "modalities": {"eeg": {"present": int(frame["eeg_path"].notna().sum()), "missing": 0}},
        "class_counts": {
            label: int(frame["diagnosis"].eq(label).sum()) for label in CLASS_NAMES
        },
        "gender_counts": {
            str(label): int(count)
            for label, count in frame["gender"].value_counts(dropna=False).items()
        },
        "missing_values": {
            "age": int(frame["age"].isna().sum()),
            "gender": int(frame["gender"].isna().sum()),
            "mmse": int(frame["mmse"].isna().sum()),
            "eeg": int(frame["eeg_path"].isna().sum()),
        },
        "recording_duration_seconds": {
            "minimum": float(frame["recording_duration_seconds"].min()),
            "median": float(frame["recording_duration_seconds"].median()),
            "maximum": float(frame["recording_duration_seconds"].max()),
        },
        "sampling_frequency_hz": sorted(
            set(frame["sampling_frequency_hz"].astype(float).tolist())
        ),
        "channels": list(CHANNELS),
        "source_group_mapping": GROUP_TO_CLASS,
    }
    return frame, inventory
