"""Build a local, non-diagnostic PDF summary for an EEG research inference."""
from __future__ import annotations

from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

CLASS_LABELS = {
    "AD": "Alzheimer's disease",
    "FTD": "Frontotemporal dementia",
    "CN": "Healthy control",
}


def _safe_text(value: Any) -> str:
    return str(value if value is not None else "Not recorded").encode(
        "ascii", errors="replace"
    ).decode("ascii")


def build_eeg_report(
    result: dict[str, Any],
    *,
    research_code: str = "",
) -> bytes:
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="NeuroFusion AD EEG Research Summary",
        author="NeuroFusion AD",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ResearchTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#143b56"),
        )
    )
    body = [
        Paragraph("NeuroFusion AD | EEG Research Summary", styles["ResearchTitle"]),
        Paragraph(
            "<b>Not a diagnosis.</b> This experimental research output is not "
            "clinically validated and must not be used for diagnosis or treatment.",
            styles["BodyText"],
        ),
        Spacer(1, 0.18 * inch),
        Paragraph("Recording and information supplied", styles["Heading2"]),
    ]
    clinical = result.get("clinical_input", {})
    preprocessing = result.get("preprocessing", {})
    input_rows = [
        ["Field", "Value"],
        ["Research code", _safe_text(research_code) if research_code else "Not provided"],
        [
            "Closest study-group match",
            _safe_text(
                CLASS_LABELS.get(
                    result.get("predicted_class"), result.get("predicted_class")
                )
            ),
        ],
        ["Age (years)", _safe_text(clinical.get("age"))],
        ["Sex recorded", _safe_text(clinical.get("gender"))],
        ["Memory-test score (MMSE)", _safe_text(clinical.get("mmse"))],
        ["Usable five-second EEG segments", _safe_text(result.get("window_count"))],
        [
            "Original recording rate (samples/second)",
            _safe_text(preprocessing.get("source_sampling_frequency_hz")),
        ],
        [
            "Model processing rate (samples/second)",
            _safe_text(preprocessing.get("sampling_frequency_hz")),
        ],
        ["Segments skipped by quality check", _safe_text(preprocessing.get("rejected_windows"))],
    ]
    body.append(_table(input_rows, (2.5 * inch, 4.5 * inch)))
    body.extend(
        [
            Spacer(1, 0.18 * inch),
            Paragraph("Model scores (not medical risk)", styles["Heading2"]),
        ]
    )
    probability_rows = [["Study group", "Model score"]]
    probability_rows.extend(
        [
            [CLASS_LABELS.get(label, label), f"{float(probability):.1%}"]
            for label, probability in result.get("class_probabilities", {}).items()
        ]
    )
    body.append(_table(probability_rows, (3.5 * inch, 3.5 * inch)))
    body.extend(
        [
            Spacer(1, 0.18 * inch),
            Paragraph("Exploratory model details", styles["Heading2"]),
            Paragraph(
                "The charts below show patterns the model responded to for this "
                "recording. They do not prove a cause or explain a person's condition.",
                styles["BodyText"],
            ),
            Spacer(1, 0.08 * inch),
        ]
    )
    attention = result.get("attention_weights", {})
    attention_matrix = attention.get("matrix", [])
    if len(attention_matrix) == 2:
        body.append(
            _table(
                [
                    ["Information weighting", "EEG", "Age / sex / MMSE"],
                    ["EEG information", *[f"{float(value):.4f}" for value in attention_matrix[0]]],
                    ["Clinical details", *[f"{float(value):.4f}" for value in attention_matrix[1]]],
                ],
                (2.0 * inch, 2.0 * inch, 2.0 * inch),
            )
        )
    channel_rows = [["EEG channel", "Relative response (not causal)"]]
    channel_rows.extend(
        [
            [name, f"{float(value):.6g}"]
            for name, value in list(result.get("channel_importance", {}).items())[:10]
        ]
    )
    if len(channel_rows) > 1:
        body.extend([Spacer(1, 0.12 * inch), _table(channel_rows, (3.5 * inch, 3.5 * inch))])
    body.extend(
        [
            Spacer(1, 0.2 * inch),
            Paragraph("Study limits", styles["Heading2"]),
            Paragraph(
                "The model was tested on a small group of 88 people from one data "
                "source. It sorts EEG recordings into existing Alzheimer's disease, "
                "frontotemporal dementia, or healthy-control study groups; it does not "
                "predict future disease or MCI-to-Alzheimer's progression. The memory "
                "test score is strongly related to the study labels. Independent "
                "validation is needed. Scores are not diagnoses or medical risk estimates.",
                styles["BodyText"],
            ),
        ]
    )
    document.build(body)
    return output.getvalue()


def _table(rows: list[list[str]], widths: tuple[float, ...]) -> Table:
    table = Table(rows, colWidths=list(widths), repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1f5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#143b56")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cad6df")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table
