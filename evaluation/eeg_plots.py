"""Create offline, interactive Plotly diagrams from measured test predictions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datasets.openneuro import CLASS_NAMES


MODEL_LABELS = {
    "multimodal": "EEG + clinical attention",
    "eeg_only": "EEG-only baseline",
    "clinical_only": "Clinical-only baseline",
}


def generate_evaluation_figures(
    results: dict[str, Any],
    output_dir: str | Path = "evaluation/figures",
) -> list[Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics = results["evaluation"]
    files: list[Path] = []

    confusion = make_subplots(
        rows=1,
        cols=len(MODEL_LABELS),
        subplot_titles=[MODEL_LABELS[key] for key in MODEL_LABELS],
        horizontal_spacing=0.1,
    )
    for column, mode in enumerate(MODEL_LABELS, start=1):
        confusion.add_trace(
            go.Heatmap(
                z=metrics[mode]["confusion_matrix"],
                x=list(CLASS_NAMES),
                y=list(CLASS_NAMES),
                text=metrics[mode]["confusion_matrix"],
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=column == 1,
                hovertemplate="True %{y}<br>Predicted %{x}<br>Participants %{z}<extra></extra>",
            ),
            row=1,
            col=column,
        )
        confusion.update_xaxes(title_text="Predicted class", row=1, col=column)
        confusion.update_yaxes(title_text="True class", autorange="reversed", row=1, col=column)
    confusion.update_layout(
        title="Held-out participant confusion matrices",
        template="plotly_white",
        height=520,
    )
    path = destination / "confusion_matrices.html"
    confusion.write_html(path, include_plotlyjs=True, full_html=True)
    files.append(path)

    comparison = go.Figure()
    for metric, title in (
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("precision_macro", "Macro precision"),
        ("recall_macro", "Macro recall"),
        ("f1_macro", "Macro F1"),
        ("auroc_ovr_macro", "AUROC (one-vs-rest)"),
        ("auprc_macro", "AUPRC (macro)"),
    ):
        values = [metrics[key].get(metric) for key in MODEL_LABELS]
        comparison.add_trace(
            go.Bar(
                name=title,
                x=[MODEL_LABELS[key] for key in MODEL_LABELS],
                y=values,
                hovertemplate=f"{title}: %{{y:.3f}}<extra></extra>",
            )
        )
    comparison.update_layout(
        title="Model comparison on held-out participants",
        yaxis={"title": "Score", "range": [0, 1]},
        barmode="group",
        template="plotly_white",
        height=560,
    )
    path = destination / "model_comparison.html"
    comparison.write_html(path, include_plotlyjs=True, full_html=True)
    files.append(path)

    roc = go.Figure()
    for mode, model_metrics in metrics.items():
        for label, curve in model_metrics["roc_curves_ovr"].items():
            roc.add_trace(
                go.Scatter(
                    x=curve["false_positive_rate"],
                    y=curve["true_positive_rate"],
                    mode="lines",
                    name=f"{MODEL_LABELS[mode]} · {label} (AUC {curve['auroc']:.2f})",
                )
            )
    roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line={"dash": "dash", "color": "#718096"},
        )
    )
    roc.update_layout(
        title="One-vs-rest ROC curves (participant-level)",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        template="plotly_white",
        height=560,
    )
    path = destination / "roc_curves.html"
    roc.write_html(path, include_plotlyjs=True, full_html=True)
    files.append(path)

    pr = go.Figure()
    for mode, model_metrics in metrics.items():
        for label, curve in model_metrics["precision_recall_curves_ovr"].items():
            pr.add_trace(
                go.Scatter(
                    x=curve["recall"],
                    y=curve["precision"],
                    mode="lines",
                    name=f"{MODEL_LABELS[mode]} · {label} (AP {curve['auprc']:.2f})",
                )
            )
    pr.update_layout(
        title="One-vs-rest precision-recall curves (participant-level)",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        template="plotly_white",
        height=560,
    )
    path = destination / "precision_recall_curves.html"
    pr.write_html(path, include_plotlyjs=True, full_html=True)
    files.append(path)

    calibration = go.Figure()
    for mode, model_metrics in metrics.items():
        bins = model_metrics["calibration"]["reliability_bins"]
        if bins:
            calibration.add_trace(
                go.Scatter(
                    x=[item["confidence"] for item in bins],
                    y=[item["accuracy"] for item in bins],
                    mode="lines+markers",
                    name=MODEL_LABELS[mode],
                    text=[item["count"] for item in bins],
                    hovertemplate="Confidence %{x:.2f}<br>Accuracy %{y:.2f}"
                    "<br>Participants %{text}<extra></extra>",
                )
            )
    calibration.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Ideal",
            line={"dash": "dash", "color": "#718096"},
        )
    )
    calibration.update_layout(
        title="Confidence calibration (10-bin, participant-level)",
        xaxis_title="Mean predicted confidence",
        yaxis_title="Observed accuracy",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        template="plotly_white",
        height=560,
    )
    path = destination / "calibration.html"
    calibration.write_html(path, include_plotlyjs=True, full_html=True)
    files.append(path)

    report_path = destination / "index.html"
    links = "\n".join(
        f'<li><a href="{file.name}">{file.stem.replace("_", " ").title()}</a></li>'
        for file in files
    )
    report_path.write_text(
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>NeuroFusion AD | Held-out EEG evaluation</title>"
        "<style>body{font:16px/1.5 Segoe UI,Arial,sans-serif;max-width:900px;"
        "margin:3rem auto;padding:0 1rem;color:#17324d}a{color:#087e8b}"
        "li{margin:.5rem 0}.note{background:#fff4d6;padding:1rem;border-radius:8px}"
        "</style><h1>NeuroFusion AD: ds004504 held-out evaluation</h1>"
        "<p>Interactive diagrams generated from participant-level held-out "
        "predictions in evaluation/metrics.json.</p><ul>"
        f"{links}</ul><p class=\"note\">Cross-sectional AD/FTD/CN classification only. "
        "The 88-person cohort is small; MMSE is associated with diagnosis; "
        "no external validation was performed. Research output only, not a diagnosis.</p>"
        "</html>",
        encoding="utf-8",
    )
    files.append(report_path)

    measured_rows = []
    for mode, label in MODEL_LABELS.items():
        values = metrics[mode]
        ci = values["stratified_bootstrap_95_ci"]["macro_f1_95_ci"]
        ci_text = "not estimable" if ci is None else f"{ci[0]:.3f}–{ci[1]:.3f}"
        measured_rows.append(
            "<tr>"
            f"<td>{label}</td>"
            f"<td>{values['participant_count']}</td>"
            f"<td>{values['accuracy']:.3f}</td>"
            f"<td>{values['balanced_accuracy']:.3f}</td>"
            f"<td>{values['f1_macro']:.3f} ({ci_text})</td>"
            f"<td>{_optional_metric(values['auroc_ovr_macro'])}</td>"
            f"<td>{_optional_metric(values['auprc_macro'])}</td>"
            f"<td>{values['calibration']['brier_score']:.3f}</td>"
            f"<td>{values['calibration']['expected_calibration_error_10_bins']:.3f}</td>"
            "</tr>"
        )
    finding_report = Path("reports/validation_findings.html")
    finding_report.parent.mkdir(parents=True, exist_ok=True)
    finding_report.write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>NeuroFusion AD | EEG Evaluation Findings</title>"
        "<style>body{font:16px/1.55 Segoe UI,Arial,sans-serif;max-width:1180px;"
        "margin:2rem auto;padding:0 1rem;color:#17324d}header{background:#102a43;"
        "color:white;padding:1.5rem 2rem;border-radius:14px}section{margin:1rem 0;"
        "padding:1.2rem;border:1px solid #d7e1e9;border-radius:12px}table{width:100%;"
        "border-collapse:collapse}th,td{text-align:left;padding:.6rem;border-bottom:"
        "1px solid #d7e1e9}th{background:#f3f7fa}.warn{background:#fff7df;padding:1rem;"
        "border-left:5px solid #d89c27}a{color:#087e8b}</style></head><body>"
        "<header><h1>NeuroFusion AD | EEG Evaluation Findings</h1>"
        "<p>OpenNeuro ds004504 · participant-level held-out evaluation</p></header>"
        "<section><h2>Measured held-out comparison</h2>"
        f"<p>Test participants: {results['split']['participant_counts']['test']}; "
        f"test class counts: {results['split']['test_class_counts']}.</p>"
        "<table><thead><tr><th>Model</th><th>N</th><th>Accuracy</th><th>Balanced "
        "accuracy</th><th>Macro F1 (95% bootstrap CI)</th><th>Macro AUROC</th>"
        "<th>Macro AUPRC</th><th>Brier</th><th>10-bin ECE</th></tr></thead><tbody>"
        + "".join(measured_rows)
        + "</tbody></table><p>All values are generated from saved model weights and "
        "one mean-window probability vector per held-out participant. See "
        "<a href=\"../evaluation/metrics.json\">the measured metrics JSON</a>.</p></section>"
        "<section><h2>Offline interactive diagrams</h2><ul>"
        "<li><a href=\"../evaluation/figures/confusion_matrices.html\">Confusion matrices</a></li>"
        "<li><a href=\"../evaluation/figures/model_comparison.html\">Model comparison</a></li>"
        "<li><a href=\"../evaluation/figures/roc_curves.html\">ROC curves</a></li>"
        "<li><a href=\"../evaluation/figures/precision_recall_curves.html\">Precision-recall curves</a></li>"
        "<li><a href=\"../evaluation/figures/calibration.html\">Calibration diagram</a></li>"
        "</ul></section><section class=\"warn\"><b>Interpretation limits:</b> this is "
        "cross-sectional AD/FTD/CN classification, not MCI-to-AD progression. The "
        "88-participant cohort is small and single-source; MMSE is strongly associated "
        "with the group labels; no external validation was performed. Bootstrap "
        "intervals are participant-level uncertainty summaries, not evidence of "
        "clinical validity. Attention and gradient outputs are not causal explanations."
        "</section></body></html>",
        encoding="utf-8",
    )
    files.append(finding_report)
    return files


def _optional_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> None:
    metrics_path = Path("evaluation/metrics.json")
    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    for path in generate_evaluation_figures(results):
        print(path)


if __name__ == "__main__":
    main()
