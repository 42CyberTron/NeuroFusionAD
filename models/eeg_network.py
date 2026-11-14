"""Compact EEG CNN, clinical encoder and masked cross-modal attention classifier."""
from __future__ import annotations

import torch
from torch import nn


class EEGEncoder(nn.Module):
    def __init__(self, channels: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(96, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(eeg))


class EEGClinicalNet(nn.Module):
    """Classify AD/FTD/CN using EEG, clinical features, or both."""

    MODES = {"multimodal", "eeg_only", "clinical_only"}

    def __init__(
        self,
        clinical_features: int = 7,
        channels: int = 19,
        hidden_size: int = 64,
        num_classes: int = 3,
        dropout: float = 0.25,
        mode: str = "multimodal",
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}.")
        if hidden_size % 4:
            raise ValueError("hidden_size must be divisible by 4 attention heads.")
        self.mode = mode
        self.eeg_encoder = EEGEncoder(channels, hidden_size, dropout)
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.modality_embedding = nn.Parameter(torch.zeros(1, 2, hidden_size))
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, dropout=dropout, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(
        self,
        eeg: torch.Tensor,
        clinical: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if eeg.ndim != 3 or clinical.ndim != 2 or modality_mask.shape != (eeg.shape[0], 2):
            raise ValueError("Expected EEG [batch, channels, time], clinical [batch, features], "
                             "and a [batch, 2] modality mask.")
        present = modality_mask.bool().clone()
        if self.mode == "eeg_only":
            present[:, 1] = False
        elif self.mode == "clinical_only":
            present[:, 0] = False

        if self.mode == "clinical_only":
            eeg_token = eeg.new_zeros((len(eeg), self.modality_embedding.shape[-1]))
        else:
            eeg_token = self.eeg_encoder(eeg)
        if self.mode == "eeg_only":
            clinical_token = clinical.new_zeros((len(clinical), self.modality_embedding.shape[-1]))
        else:
            clinical_token = self.clinical_encoder(clinical)
        tokens = torch.stack((eeg_token, clinical_token), dim=1)
        tokens = tokens + self.modality_embedding

        no_modalities = ~present.any(dim=1, keepdim=True)
        safe_present = present.clone()
        safe_present[:, 0:1] |= no_modalities
        attended, weights = self.attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=~safe_present,
            need_weights=True,
            average_attn_weights=False,
        )
        query_mask = present.unsqueeze(-1).to(attended.dtype)
        fused = (attended * query_mask).sum(dim=1)
        fused = fused / query_mask.sum(dim=1).clamp_min(1.0)
        fused = self.fusion_norm(fused)
        fused = torch.where(no_modalities, torch.zeros_like(fused), fused)
        return {
            "logits": self.classifier(fused),
            "attention_weights": weights,
            "eeg_embedding": eeg_token,
            "clinical_embedding": clinical_token,
            "present_modalities": present,
        }
