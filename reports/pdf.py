"""Generate a PDF report from actual inference output."""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def build_patient_report(
    patient: dict[str, Any],
    visits: list[dict[str, Any]],
    prediction: dict[str, Any],
) -> bytes:
    output = BytesIO()
    document = SimpleDocTemplate(output, pagesize=letter, title="NeuroFusion AD Research Report")
    styles = getSampleStyleSheet()
    story = [
        Paragraph("NeuroFusion AD — Research Prediction Report", styles["Title"]),
        Paragraph("Not a clinical diagnosis. Research use only.", styles["Heading2"]),
        Paragraph(
            f"Patient: {escape(str(patient['display_name']))} "
            f"({escape(str(patient['patient_id']))})<br/>"
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}<br/>"
            f"Visits included: {len(visits)}",
            styles["BodyText"],
        ),
        Spacer(1, 14),
        Paragraph("Model output", styles["Heading2"]),
    ]
    diagnosis = prediction["diagnosis"]
    story.append(
        Table(
            [["Class", "Probability"]]
            + [[label, f"{value:.1%}"] for label, value in diagnosis.items()],
            hAlign="LEFT",
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#18324A")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            ),
        )
    )
    story.extend([Spacer(1, 10), Paragraph("Progression estimates", styles["Heading2"])])
    story.append(
        Table(
            [["Horizon", "Estimated probability"]]
            + [
                [horizon.replace("_", " "), f"{value:.1%}"]
                for horizon, value in prediction["progression"].items()
            ],
            hAlign="LEFT",
        )
    )
    story.extend(
        [
            Spacer(1, 10),
            Paragraph("Modality ablation deltas (probability-point change)", styles["Heading2"]),
            Paragraph(str(prediction["modality_ablation_delta"]), styles["BodyText"]),
            Paragraph("Clinical integrated-gradients attribution (not causal)", styles["Heading2"]),
            Paragraph(str(prediction["clinical_feature_gradient_attribution"]), styles["BodyText"]),
            Spacer(1, 10),
            Paragraph("Visit input summary", styles["Heading2"]),
        ]
    )
    for visit in visits:
        modalities = []
        if visit.get("mri_path"):
            modalities.append("MRI")
        if visit.get("pet_path"):
            modalities.append("PET")
        story.append(
            Paragraph(
                f"{escape(str(visit['visit_date']))}: clinical features "
                f"{escape(', '.join(visit.get('clinical', {}).keys()) or 'none')}; "
                f"imaging {', '.join(modalities) or 'none'}.",
                styles["BodyText"],
            )
        )
    if prediction.get("mri_gradcam") is not None:
        import numpy as np

        cam = np.asarray(prediction["mri_gradcam"], dtype=np.float32)
        image_buffer = BytesIO()
        from PIL import Image as PILImage

        PILImage.fromarray(np.uint8(np.clip(cam, 0, 1) * 255)).save(image_buffer, format="PNG")
        image_buffer.seek(0)
        story.extend([Spacer(1, 10), Paragraph("MRI Grad-CAM", styles["Heading2"])])
        story.append(Image(image_buffer, width=220, height=220))
    story.append(
        Paragraph(
            "The output depends on the local trained checkpoint and provided inputs. "
            "It is not validated for clinical decision-making and must not be used "
            "to diagnose or guide treatment.",
            styles["Italic"],
        )
    )
    document.build(story)
    return output.getvalue()
