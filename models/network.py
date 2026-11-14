"""Missingness-aware multimodal longitudinal prediction network."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ImageEncoder(nn.Module):
    """Small 2D CNN for preprocessed single-channel structural/functional images."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.projection = nn.Linear(96, hidden_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(image))


class NeuroFusionNet(nn.Module):
    """Fuse clinical, MRI and PET tokens, then model visits with a time-aware GRU."""

    def __init__(
        self,
        clinical_features: int = 10,
        hidden_size: int = 128,
        num_diagnoses: int = 3,
        num_horizons: int = 3,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.mri_encoder = ImageEncoder(hidden_size)
        self.pet_encoder = ImageEncoder(hidden_size)
        self.modality_embedding = nn.Parameter(torch.zeros(1, 3, hidden_size))
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, dropout=dropout, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(hidden_size)
        self.time_projection = nn.Sequential(
            nn.Linear(1, hidden_size), nn.Tanh()
        )
        self.temporal = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.diagnosis_head = nn.Linear(hidden_size, num_diagnoses)
        self.progression_head = nn.Linear(hidden_size, num_horizons)

    def forward(
        self,
        clinical: torch.Tensor,
        mri: torch.Tensor,
        pet: torch.Tensor,
        modality_mask: torch.Tensor,
        intervals_days: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Inputs are [batch, visits, ...]; imaging masks are false for absent scans.
        batch, visits, _, height, width = mri.shape
        clinical_token = self.clinical_encoder(clinical)
        flat_mask = modality_mask.reshape(batch * visits, 3).bool()
        mri_token = self._encode_present(
            self.mri_encoder, mri.reshape(batch * visits, 1, height, width), flat_mask[:, 1]
        )
        pet_token = self._encode_present(
            self.pet_encoder, pet.reshape(batch * visits, 1, height, width), flat_mask[:, 2]
        )
        tokens = torch.stack(
            (
                clinical_token,
                mri_token.reshape(batch, visits, -1),
                pet_token.reshape(batch, visits, -1),
            ),
            dim=2,
        ) + self.modality_embedding

        present = modality_mask.bool()
        # Avoid an all-masked attention row. The resulting zero representation is
        # still passed through the model, while inference separately rejects empty data.
        no_modalities = ~present.any(dim=-1, keepdim=True)
        safe_present = present.clone()
        safe_present[..., 0:1] |= no_modalities
        attended, _ = self.attention(
            tokens.reshape(batch * visits, 3, -1),
            tokens.reshape(batch * visits, 3, -1),
            tokens.reshape(batch * visits, 3, -1),
            key_padding_mask=~safe_present.reshape(batch * visits, 3),
            need_weights=False,
        )
        attended = attended.reshape(batch, visits, 3, -1)
        weights = safe_present.unsqueeze(-1).to(attended.dtype)
        fused = (attended * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        fused = self.fusion_norm(fused)
        fused = torch.where(no_modalities, torch.zeros_like(fused), fused)
        elapsed = torch.log1p(intervals_days.clamp_min(0) / 30.0).unsqueeze(-1)
        temporal_input = fused + self.time_projection(elapsed)
        lengths = visit_mask.long().sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(
            temporal_input, lengths, batch_first=True, enforce_sorted=False
        )
        packed_sequence, _ = self.temporal(packed)
        sequence, _ = pad_packed_sequence(
            packed_sequence, batch_first=True, total_length=visits
        )
        final_indices = lengths.to(sequence.device) - 1
        final = self.dropout(sequence[torch.arange(batch, device=sequence.device), final_indices])
        return {
            "diagnosis_logits": self.diagnosis_head(final),
            "progression_logits": self.progression_head(final),
        }

    @staticmethod
    def _encode_present(
        encoder: ImageEncoder, images: torch.Tensor, present: torch.Tensor
    ) -> torch.Tensor:
        indices = present.nonzero(as_tuple=False).flatten()
        encoded = images.new_zeros((images.shape[0], encoder.projection.out_features))
        if indices.numel():
            selected = encoder(images.index_select(0, indices))
            encoded = encoded.index_copy(0, indices, selected)
        return encoded
