"""Grad-CAM postprocessing for the final MRI CNN feature map."""
from __future__ import annotations

import numpy as np
import torch


def normalize_gradcam(
    activation: torch.Tensor, gradient: torch.Tensor, image_size: int, flat_index: int
) -> np.ndarray:
    if activation.ndim != 4 or gradient.shape != activation.shape:
        raise ValueError("Grad-CAM feature maps and gradients must have shape [N,C,H,W].")
    if flat_index < 0 or flat_index >= activation.shape[0]:
        raise IndexError("Grad-CAM visit index is outside the feature-map batch.")
    weights = gradient[flat_index].mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * activation[flat_index]).sum(dim=0))
    cam = torch.nn.functional.interpolate(
        cam[None, None],
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    cam = cam / cam.max().clamp_min(1e-8)
    return cam.detach().cpu().numpy()
