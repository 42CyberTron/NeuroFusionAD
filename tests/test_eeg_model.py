import torch

from models.eeg_network import EEGClinicalNet


def test_attention_fusion_uses_both_modalities_and_masks_absent_clinical():
    torch.manual_seed(7)
    model = EEGClinicalNet(hidden_size=32, dropout=0)
    model.eval()
    eeg = torch.randn(2, 19, 640)
    clinical = torch.randn(2, 7, requires_grad=True)
    mask = torch.tensor([[True, True], [True, False]])
    output = model(eeg, clinical, mask)

    assert output["logits"].shape == (2, 3)
    assert output["attention_weights"].shape == (2, 4, 2, 2)
    assert torch.isfinite(output["logits"]).all()
    output["logits"][0, 0].backward()
    assert clinical.grad is not None
    assert clinical.grad[0].abs().sum() > 0
    assert output["present_modalities"].tolist() == [[True, True], [True, False]]


def test_single_modality_baselines_produce_three_class_logits():
    eeg = torch.randn(2, 19, 640)
    clinical = torch.randn(2, 7)
    mask = torch.ones(2, 2, dtype=torch.bool)
    for mode in ("eeg_only", "clinical_only"):
        model = EEGClinicalNet(hidden_size=32, dropout=0, mode=mode)
        result = model(eeg, clinical, mask)
        assert result["logits"].shape == (2, 3)
        assert torch.isfinite(result["logits"]).all()
