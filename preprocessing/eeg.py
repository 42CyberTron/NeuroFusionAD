"""MNE-based preprocessing for the original continuous ds004504 EEG recordings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from datasets.openneuro import CHANNELS


TARGET_SFREQ = 128
LOW_CUTOFF_HZ = 0.5
HIGH_CUTOFF_HZ = 45.0
WINDOW_SECONDS = 5
ARTIFACT_PEAK_TO_PEAK_UV = 500.0
ARTIFACT_ABSOLUTE_PEAK_UV = 500.0


@dataclass(frozen=True)
class ProcessedRecording:
    windows_uv: np.ndarray
    channel_names: tuple[str, ...]
    original_sfreq: float
    duration_seconds: float
    rejected_windows: int
    total_windows: int


def read_raw_eeg(path: str | Path, *, preload: bool = True) -> mne.io.BaseRaw:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"EEG recording does not exist: {source}")
    if source.stat().st_size == 0:
        raise ValueError("EEG recording is empty.")
    if source.suffix.lower() != ".set":
        raise ValueError("Only EEGLAB .set recordings are supported.")
    try:
        return mne.io.read_raw_eeglab(
            str(source), preload=preload, verbose="ERROR"
        )
    except Exception as exc:
        raise ValueError(f"MNE could not read the EEGLAB recording: {source.name}") from exc


def preprocess_raw(
    raw: mne.io.BaseRaw,
    *,
    window_seconds: int = WINDOW_SECONDS,
) -> ProcessedRecording:
    """Filter, resample, order 19 channels and reject gross-amplitude windows."""
    if window_seconds < 1:
        raise ValueError("window_seconds must be a positive integer.")
    working = raw.copy().load_data()
    if not np.isfinite(working.info["sfreq"]) or working.info["sfreq"] <= HIGH_CUTOFF_HZ * 2:
        raise ValueError("EEG sampling frequency must exceed 90 Hz.")

    normalized_names = {name.casefold(): name for name in working.ch_names}
    if len(normalized_names) != len(working.ch_names):
        raise ValueError("EEG recording contains duplicate channel names.")
    missing = [name for name in CHANNELS if name.casefold() not in normalized_names]
    if missing:
        raise ValueError(f"EEG recording is missing required channels: {missing}")

    rename = {
        normalized_names[name.casefold()]: name
        for name in CHANNELS
        if normalized_names[name.casefold()] != name
    }
    if rename:
        working.rename_channels(rename)
    working.pick(list(CHANNELS))
    if tuple(working.ch_names) != CHANNELS:
        raise ValueError("Could not establish the required canonical channel order.")

    original_sfreq = float(working.info["sfreq"])
    duration_seconds = float(working.n_times / original_sfreq)
    working.filter(
        l_freq=LOW_CUTOFF_HZ,
        h_freq=HIGH_CUTOFF_HZ,
        method="iir",
        iir_params={"order": 4, "ftype": "butter", "output": "sos"},
        verbose="ERROR",
    )
    working.resample(TARGET_SFREQ, npad="auto", verbose="ERROR")

    signal_uv = working.get_data(picks=list(CHANNELS)) * 1_000_000.0
    if signal_uv.ndim != 2 or not np.isfinite(signal_uv).all():
        raise ValueError("Filtered EEG contains non-finite signal values.")
    if float(np.std(signal_uv)) < 0.05:
        raise ValueError("EEG recording is effectively flat after preprocessing.")

    window_samples = TARGET_SFREQ * window_seconds
    total_samples = signal_uv.shape[1] // window_samples * window_samples
    if total_samples < window_samples:
        raise ValueError(
            f"EEG recording must contain at least one complete {window_seconds}-second window."
        )
    windows = signal_uv[:, :total_samples].reshape(
        len(CHANNELS), -1, window_samples
    ).transpose(1, 0, 2)
    peak_to_peak = np.ptp(windows, axis=2).max(axis=1)
    absolute_peak = np.abs(windows).max(axis=(1, 2))
    keep = (
        np.isfinite(windows).all(axis=(1, 2))
        & (peak_to_peak <= ARTIFACT_PEAK_TO_PEAK_UV)
        & (absolute_peak <= ARTIFACT_ABSOLUTE_PEAK_UV)
    )
    accepted = np.asarray(windows[keep], dtype=np.float32)
    if not len(accepted):
        raise ValueError("No EEG windows passed finite-signal and amplitude checks.")
    return ProcessedRecording(
        windows_uv=accepted,
        channel_names=CHANNELS,
        original_sfreq=original_sfreq,
        duration_seconds=duration_seconds,
        rejected_windows=int((~keep).sum()),
        total_windows=int(len(windows)),
    )


def preprocess_eeg_file(path: str | Path, *, window_seconds: int = WINDOW_SECONDS) -> ProcessedRecording:
    return preprocess_raw(read_raw_eeg(path), window_seconds=window_seconds)
