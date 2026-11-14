import mne
import numpy as np
import pytest

from datasets.openneuro import CHANNELS
from preprocessing.eeg import preprocess_raw


def _raw(channels=CHANNELS):
    rng = np.random.default_rng(5)
    data = rng.normal(0, 10e-6, (len(channels), 5000))
    info = mne.create_info(list(channels), sfreq=500.0, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="ERROR")


def test_mne_pipeline_filters_resamples_and_segments_software_test_signal():
    result = preprocess_raw(_raw(), window_seconds=5)
    assert result.windows_uv.shape == (2, 19, 640)
    assert result.channel_names == CHANNELS
    assert result.original_sfreq == 500
    assert result.rejected_windows == 0
    assert np.isfinite(result.windows_uv).all()


def test_mne_pipeline_rejects_recording_with_missing_required_channels():
    with pytest.raises(ValueError, match="missing required channels"):
        preprocess_raw(_raw(CHANNELS[:18]))
