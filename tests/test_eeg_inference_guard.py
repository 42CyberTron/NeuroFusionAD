import pytest

from inference.eeg_predictor import EEGPredictor


def test_eeg_inference_requires_trained_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="No trained EEG checkpoint"):
        EEGPredictor(tmp_path / "missing.pt", device="cpu")
