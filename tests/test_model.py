"""Unit tests for tensor mechanics; these tensors are not patient/image data."""
import torch

from models.network import NeuroFusionNet


def test_model_accepts_missing_modalities_and_multiple_visits():
    model = NeuroFusionNet(
        clinical_features=4,
        hidden_size=32,
        num_diagnoses=3,
        num_horizons=2,
        dropout=0,
    )
    clinical = torch.zeros(2, 3, 4)
    mri = torch.zeros(2, 3, 1, 64, 64)
    pet = torch.zeros_like(mri)
    mask = torch.zeros(2, 3, 3, dtype=torch.bool)
    mask[0, :, 0] = True
    mask[1, 0, 1] = True
    visits = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    result = model(
        clinical,
        mri,
        pet,
        mask,
        torch.tensor([[0.0, 30.0, 90.0], [0.0, 0.0, 0.0]]),
        visits,
    )
    assert result["diagnosis_logits"].shape == (2, 3)
    assert result["progression_logits"].shape == (2, 2)
    assert torch.isfinite(result["progression_logits"]).all()


def test_model_masks_missing_clinical_and_pet_tokens():
    model = NeuroFusionNet(clinical_features=2, hidden_size=32, dropout=0)
    model.eval()
    inputs = (
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 1, 64, 64),
        torch.zeros(1, 1, 1, 64, 64),
        torch.tensor([[[False, True, False]]]),
        torch.zeros(1, 1),
        torch.ones(1, 1, dtype=torch.bool),
    )
    with torch.no_grad():
        result = model(*inputs)
    assert torch.isfinite(result["diagnosis_logits"]).all()
