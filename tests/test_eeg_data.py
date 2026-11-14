import numpy as np
import pandas as pd
import pytest

from datasets.openneuro import CLASS_NAMES
from training.eeg_data import create_participant_split, encode_clinical


def test_participant_split_is_disjoint_complete_and_stratified():
    labels = ["AD"] * 36 + ["FTD"] * 23 + ["CN"] * 29
    frame = pd.DataFrame(
        {
            "participant_id": [f"sub-{index:03d}" for index in range(1, 89)],
            "diagnosis": labels,
        }
    )
    split = create_participant_split(frame, seed=42)
    partitions = [set(values) for values in split.values()]
    assert not (partitions[0] & partitions[1])
    assert not (partitions[0] & partitions[2])
    assert not (partitions[1] & partitions[2])
    assert set.union(*partitions) == set(frame["participant_id"])
    assert all(len(values) > 0 for values in split.values())


def test_clinical_encoder_tracks_missingness_and_validates_ranges():
    scaling = {
        "age": {"mean": 65.0, "scale": 8.0},
        "mmse": {"mean": 22.0, "scale": 7.0},
    }
    encoded, present = encode_clinical(
        {"age": None, "gender": "Other", "mmse": 30}, scaling
    )
    assert present
    assert encoded.tolist() == pytest.approx([0.0, 8.0 / 7.0, 0, 0, 1, 0, 1])
    with pytest.raises(ValueError, match="between 18 and 120"):
        encode_clinical({"age": 17, "gender": "M", "mmse": 25}, scaling)
    assert set(CLASS_NAMES) == {"AD", "FTD", "CN"}


@pytest.mark.parametrize(
    ("age", "mmse"),
    [(18, 0), (120, 30)],
)
def test_clinical_encoder_accepts_documented_input_range_endpoints(age, mmse):
    scaling = {
        "age": {"mean": 65.0, "scale": 8.0},
        "mmse": {"mean": 22.0, "scale": 7.0},
    }

    encoded, present = encode_clinical(
        {"age": age, "gender": "Female", "mmse": mmse}, scaling
    )

    assert present
    assert np.isfinite(encoded).all()


@pytest.mark.parametrize(
    ("field", "value"),
    [("age", 121), ("mmse", -1), ("mmse", 31)],
)
def test_clinical_encoder_rejects_values_outside_documented_ranges(field, value):
    scaling = {
        "age": {"mean": 65.0, "scale": 8.0},
        "mmse": {"mean": 22.0, "scale": 7.0},
    }
    record = {"age": 65, "gender": "Female", "mmse": 22}
    record[field] = value

    with pytest.raises(ValueError):
        encode_clinical(record, scaling)
