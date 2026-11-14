import torch
import pytest

from inference.predictor import NeuroFusionPredictor


def test_inference_requires_a_trained_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train on an authorized cohort"):
        NeuroFusionPredictor(tmp_path / "missing.pt")


def test_inference_rejects_checkpoint_with_numeric_sex_feature(tmp_path):
    checkpoint_path = tmp_path / "scalar_sex.pt"
    torch.save(
        {
            "trained": True,
            "state_dict": {"unused": torch.tensor(1)},
            "config": {"clinical_features": ["sex"]},
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="scalar sex input"):
        NeuroFusionPredictor(checkpoint_path, device="cpu")
