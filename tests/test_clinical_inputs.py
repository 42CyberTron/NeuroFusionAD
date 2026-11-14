import json
from pathlib import Path

import pandas as pd
import pytest

from datasets.manifest import load_manifest
from preprocessing.clinical import SEX_FEATURES, canonicalize_clinical, encode_sex


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Male", {"sex_male": 1.0, "sex_female": 0.0, "sex_other": 0.0}),
        ("Female", {"sex_male": 0.0, "sex_female": 1.0, "sex_other": 0.0}),
        ("Other", {"sex_male": 0.0, "sex_female": 0.0, "sex_other": 1.0}),
        ("Not recorded", {"sex_male": None, "sex_female": None, "sex_other": None}),
    ],
)
def test_sex_is_categorical_and_encoded_as_one_hot(value, expected):
    assert encode_sex(value) == expected


def test_sex_requires_codebook_mapping_for_unrecognized_codes():
    with pytest.raises(ValueError, match="map source codes using its codebook"):
        encode_sex("0")


def test_model_config_uses_categorical_sex_features():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    features = json.loads(config_path.read_text(encoding="utf-8"))["clinical_features"]

    assert "sex" not in features
    assert set(SEX_FEATURES).issubset(features)


def test_sex_one_hot_fields_must_form_one_complete_category():
    with pytest.raises(ValueError, match="exactly one 1"):
        canonicalize_clinical(
            {"sex_male": 1, "sex_female": 1, "sex_other": 0}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mmse", 31),
        ("mmse", 26.5),
        ("adas13", -1),
        ("cdrsb", 19),
        ("faq", 31),
        ("faq", 12.5),
        ("apoe4", 1.5),
        ("hippocampal_volume", 0),
    ],
)
def test_invalid_clinical_measurements_are_rejected(field, value):
    with pytest.raises(ValueError):
        canonicalize_clinical({field: value})


def test_manifest_encodes_categorical_sex_and_rejects_source_codes(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"patient_id": "p1", "visit_date": "2024-01-01", "diagnosis": "MCI", "sex": "Female"},
            {"patient_id": "p2", "visit_date": "2024-01-01", "diagnosis": "MCI", "sex": "Other"},
        ]
    ).to_csv(manifest_path, index=False)
    frame = load_manifest(manifest_path)

    assert frame["sex_female"].tolist() == [1.0, 0.0]
    assert frame["sex_other"].tolist() == [0.0, 1.0]
    assert "sex" not in frame.columns

    pd.DataFrame(
        [
            {"patient_id": "p1", "visit_date": "2024-01-01", "diagnosis": "MCI", "sex": "0"},
            {"patient_id": "p2", "visit_date": "2024-01-01", "diagnosis": "MCI", "sex": "1"},
        ]
    ).to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match="map source codes using its codebook"):
        load_manifest(manifest_path)
