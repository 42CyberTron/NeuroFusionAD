import pytest

from preprocessing.imaging import validate_image_path


def test_imaging_validator_rejects_missing_and_unsupported_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_image_path(tmp_path / "missing.nii")
    unsupported = tmp_path / "scan.png"
    unsupported.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="NIfTI"):
        validate_image_path(unsupported)


def test_imaging_validator_rejects_empty_nifti(tmp_path):
    empty = tmp_path / "empty.nii"
    empty.touch()
    with pytest.raises(ValueError, match="empty"):
        validate_image_path(empty)
