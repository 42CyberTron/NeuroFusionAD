"""Canonical clinical-field encoding and validation shared by UI and training."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd


SEX_FEATURES = ("sex_male", "sex_female", "sex_other")
SEX_ALIASES = {
    "m": "male",
    "male": "male",
    "man": "male",
    "f": "female",
    "female": "female",
    "woman": "female",
    "o": "other",
    "other": "other",
    "non-binary": "other",
    "nonbinary": "other",
    "non binary": "other",
    "unknown": None,
    "not recorded": None,
    "not reported": None,
    "prefer not to say": None,
    "undisclosed": None,
}
CLINICAL_RANGES = {
    "age": (18.0, 120.0),
    "education_years": (0.0, 40.0),
    "mmse": (0.0, 30.0),
    "adas13": (0.0, 85.0),
    "cdrsb": (0.0, 18.0),
    "faq": (0.0, 30.0),
    "apoe4": (0.0, 2.0),
    "hippocampal_volume": (0.0, None),
    "brain_volume": (0.0, None),
}
INTEGER_CLINICAL_FEATURES = {"mmse", "faq", "apoe4"}


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value)) or (
            isinstance(value, str) and not value.strip()
        )
    except (TypeError, ValueError):
        return False


def encode_sex(value: Any) -> dict[str, float | None]:
    if value is None:
        category = None
    else:
        normalized = str(value).strip().lower()
        if not normalized:
            category = None
        elif normalized not in SEX_ALIASES:
            choices = "Male, Female, Other, or Not recorded"
            raise ValueError(f"Sex must be {choices}; map source codes using its codebook.")
        else:
            category = SEX_ALIASES[normalized]
    return {
        feature: (None if category is None else float(feature == f"sex_{category}"))
        for feature in SEX_FEATURES
    }


def canonicalize_clinical(values: dict[str, Any]) -> dict[str, Any]:
    """Convert a categorical sex value to model features and validate numeric fields."""
    result = dict(values)
    if "sex" in result:
        if any(feature in result for feature in SEX_FEATURES):
            raise ValueError("Provide sex as a category or one-hot fields, not both.")
        encoded = encode_sex(result.pop("sex"))
        result.update(encoded)
    validate_numeric_clinical(result)
    return result


def validate_numeric_clinical(values: dict[str, Any]) -> None:
    for feature, (minimum, maximum) in CLINICAL_RANGES.items():
        value = values.get(feature)
        if _is_missing(value):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{feature} must be numeric.") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{feature} must be finite.")
        if (minimum is not None and numeric < minimum) or (
            maximum is not None and numeric > maximum
        ):
            range_text = (
                f"{minimum:g} or greater"
                if maximum is None
                else f"between {minimum:g} and {maximum:g}"
            )
            raise ValueError(f"{feature} must be {range_text}.")
        if feature in INTEGER_CLINICAL_FEATURES and not numeric.is_integer():
            if feature == "apoe4":
                raise ValueError("apoe4 must be an integer allele count (0, 1, or 2).")
            raise ValueError(f"{feature} must be an integer score.")
        if feature in ("hippocampal_volume", "brain_volume") and numeric == 0:
            raise ValueError(
                f"{feature} must be positive; leave it blank if it is unavailable."
            )

    present_sex = [values.get(feature) for feature in SEX_FEATURES]
    if any(not _is_missing(value) for value in present_sex):
        if any(_is_missing(value) for value in present_sex):
            raise ValueError("All three sex one-hot fields must be provided together.")
        try:
            encoded = [float(value) for value in present_sex]
        except (TypeError, ValueError) as exc:
            raise ValueError("Sex one-hot fields must be numeric 0/1 values.") from exc
        if any(value not in (0.0, 1.0) for value in encoded) or sum(encoded) != 1.0:
            raise ValueError("Sex one-hot fields must contain exactly one 1 and two 0s.")
