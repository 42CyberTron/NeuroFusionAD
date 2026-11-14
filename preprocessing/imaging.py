"""Local preprocessing for authorized NIfTI volumes; no image data is fabricated."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from monai.transforms import Resize


SUPPORTED_EXTENSIONS = (".nii", ".nii.gz")


def validate_image_path(path: str | Path) -> Path:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Imaging file does not exist: {file_path}")
    if not str(file_path).lower().endswith(SUPPORTED_EXTENSIONS):
        raise ValueError("Only NIfTI files (.nii, .nii.gz) are supported.")
    if file_path.stat().st_size == 0:
        raise ValueError("Imaging file is empty.")
    return file_path


def load_nifti_slice(path: str | Path, image_size: int = 160) -> tuple[torch.Tensor, np.ndarray]:
    """Load a volume, take its central axial slice, normalize and resize."""
    file_path = validate_image_path(path)
    try:
        import SimpleITK as sitk
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("Install nibabel and SimpleITK to preprocess NIfTI imaging.") from exc

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(file_path))
    try:
        reader.ReadImageInformation()
    except RuntimeError as exc:
        raise ValueError(f"SimpleITK could not read NIfTI metadata: {file_path}") from exc
    if reader.GetDimension() != 3 or any(size < 2 for size in reader.GetSize()):
        raise ValueError("Expected a 3D NIfTI volume with at least two voxels per axis.")
    if any(not np.isfinite(spacing) or spacing <= 0 for spacing in reader.GetSpacing()):
        raise ValueError("Imaging volume contains invalid voxel spacing.")
    image = nib.as_closest_canonical(nib.load(str(file_path)))
    volume = np.asarray(image.get_fdata(dtype=np.float32))
    if volume.ndim != 3 or not np.isfinite(volume).any():
        raise ValueError("Expected a finite, non-empty 3D NIfTI volume.")
    volume = np.nan_to_num(volume, copy=False)
    axial = volume[:, :, volume.shape[2] // 2]
    low, high = np.percentile(axial, (1, 99))
    if high <= low:
        raise ValueError("Imaging volume has no usable intensity variation.")
    normalized = np.clip((axial - low) / (high - low), 0.0, 1.0)
    tensor = torch.from_numpy(normalized.copy()).float()[None]
    resized = Resize(
        spatial_size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )(tensor)
    return resized, normalized
