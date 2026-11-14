"""Preprocess verified raw BIDS recordings into participant-level window caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from datasets.openneuro import load_dataset_index
from preprocessing.eeg import (
    HIGH_CUTOFF_HZ,
    LOW_CUTOFF_HZ,
    TARGET_SFREQ,
    WINDOW_SECONDS,
    preprocess_eeg_file,
)


def process_dataset(data_root: str | Path, cache_dir: str | Path) -> Path:
    root = Path(data_root)
    destination = Path(cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frame, inventory = load_dataset_index(root)
    processed: list[dict[str, Any]] = []

    for index, row in frame.iterrows():
        participant_id = str(row["participant_id"])
        cache_path = destination / f"{participant_id}.npz"
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as existing:
                if "windows_uv" in existing and len(existing["windows_uv"]):
                    metadata = json.loads(str(existing["metadata"].item()))
                    _validate_recording_metadata(metadata, row)
                    processed.append(metadata)
                    print(f"[{index + 1:02d}/{len(frame)}] cached {participant_id}", flush=True)
                    continue
        result = preprocess_eeg_file(row["eeg_path"])
        metadata = {
            "participant_id": participant_id,
            "diagnosis": row["diagnosis"],
            "window_count": int(len(result.windows_uv)),
            "total_windows": result.total_windows,
            "rejected_windows": result.rejected_windows,
            "original_sfreq": result.original_sfreq,
            "duration_seconds": result.duration_seconds,
        }
        _validate_recording_metadata(metadata, row)
        np.savez_compressed(
            cache_path,
            windows_uv=result.windows_uv,
            metadata=np.asarray(json.dumps(metadata)),
        )
        processed.append(metadata)
        print(
            f"[{index + 1:02d}/{len(frame)}] {participant_id}: "
            f"{metadata['window_count']} windows "
            f"({metadata['rejected_windows']} amplitude/artifact windows rejected)",
            flush=True,
        )

    processed_frame = pd.DataFrame(processed)
    if len(processed_frame) != len(frame):
        raise RuntimeError("Preprocessing did not produce one cache per participant.")
    inventory.update(
        {
            "preprocessing": {
                "filter": f"4th-order Butterworth band-pass {LOW_CUTOFF_HZ}-{HIGH_CUTOFF_HZ} Hz",
                "sampling_frequency_hz": TARGET_SFREQ,
                "window_seconds": WINDOW_SECONDS,
                "artifact_rejection": (
                    "Reject 5-second windows with non-finite samples, any-channel "
                    "peak-to-peak > 500 microV, or absolute peak > 500 microV."
                ),
                "re_reference": (
                    "No additional re-reference; source sidecars describe A1/A2 "
                    "reference and contain only the 19 scalp channels."
                ),
                "ica_or_asr": (
                    "Not applied. Source documents artifact annotations as unavailable "
                    "in raw recordings; artifact handling is limited to the explicit "
                    "amplitude window rule above."
                ),
            },
            "processed_participants": int(len(processed_frame)),
            "accepted_windows": int(processed_frame["window_count"].sum()),
            "total_windows_before_rejection": int(processed_frame["total_windows"].sum()),
            "rejected_windows": int(processed_frame["rejected_windows"].sum()),
            "accepted_windows_by_class": {
                str(label): int(group["window_count"].sum())
                for label, group in processed_frame.groupby("diagnosis")
            },
            "cache_bytes": int(
                sum(path.stat().st_size for path in destination.glob("sub-*.npz"))
            ),
        }
    )
    inventory_path = destination / "dataset_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(json.dumps(inventory, indent=2), flush=True)
    return inventory_path


def _validate_recording_metadata(metadata: dict[str, Any], row: pd.Series) -> None:
    if metadata.get("participant_id") != str(row["participant_id"]):
        raise ValueError(f"Cached EEG participant identity mismatch for {row['participant_id']}.")
    if metadata.get("diagnosis") != row["diagnosis"]:
        raise ValueError(f"Cached EEG label mismatch for {row['participant_id']}.")
    if not np.isclose(
        float(metadata["original_sfreq"]),
        float(row["sampling_frequency_hz"]),
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError(f"MNE sampling rate does not match BIDS metadata for {row['participant_id']}.")
    if not np.isclose(
        float(metadata["duration_seconds"]),
        float(row["recording_duration_seconds"]),
        rtol=0,
        atol=0.02,
    ):
        raise ValueError(f"MNE duration does not match BIDS metadata for {row['participant_id']}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/openneuro_ds004504")
    parser.add_argument("--output", default="data/processed/ds004504")
    args = parser.parse_args()
    process_dataset(args.data, args.output)


if __name__ == "__main__":
    main()
