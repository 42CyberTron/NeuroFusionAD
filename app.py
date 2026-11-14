"""Offline Streamlit dashboard for EEG-based AD/FTD/CN research classification."""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from datasets.openneuro import CHANNELS, CLASS_NAMES, load_dataset_index
from reports.eeg_pdf import build_eeg_report

if TYPE_CHECKING:
    from inference.eeg_predictor import EEGPredictor


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "openneuro_ds004504"
CACHE_ROOT = ROOT / "data" / "processed" / "ds004504"
CHECKPOINT = ROOT / "checkpoints" / "eeg_multimodal.pt"
METRICS_PATH = ROOT / "evaluation" / "metrics.json"
CLASS_LABELS = {
    "AD": "Alzheimer's disease",
    "FTD": "Frontotemporal dementia",
    "CN": "Healthy control",
}

st.set_page_config(
    page_title="NeuroFusion AD | EEG Research",
    page_icon="🧠",
    layout="wide",
)
st.markdown(
    """
    <style>
      .block-container {max-width: 1440px; padding-top: 1.3rem;}
      .hero {padding: 1.6rem 2rem; border-radius: 18px; color: white;
        background: linear-gradient(120deg,#102a43,#176b87); margin-bottom: 1rem;}
      .hero h1 {margin: 0; font-size: 2rem;}
      .hero p {opacity: .9; margin: .4rem 0 0;}
      .disclaimer {border-left: 5px solid #d89c27; padding: .8rem 1rem;
        color: #f3f6f9; background: #263746; border-radius: 6px;}
      .disclaimer b {color: #ffffff;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>NeuroFusion AD</h1>'
    "<p>Research tool for comparing EEG recordings with three study groups</p></div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="disclaimer"><b>Research use only.</b> This experimental model is '
    "not clinically validated and must not be used to diagnose or guide treatment. "
    "The dataset compares existing study groups; it does not predict future disease.</div>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Dataset",
        "View EEG",
        "New prediction",
        "Why this result?",
        "Model results",
        "About this study",
    ],
)
st.sidebar.caption("EEG recordings and inference remain on this computer.")
st.sidebar.caption("Source: public research dataset (OpenNeuro ds004504, CC0).")


def load_cohort() -> tuple[pd.DataFrame, dict] | None:
    try:
        return _load_cohort_cached(str(DATA_ROOT))
    except FileNotFoundError as exc:
        st.info(str(exc))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not validate the local dataset: {exc}")
    return None


@st.cache_data(ttl=60, show_spinner=False)
def _load_cohort_cached(root: str) -> tuple[pd.DataFrame, dict]:
    return load_dataset_index(root)


def load_metrics() -> dict | None:
    if not METRICS_PATH.is_file():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read held-out evaluation results: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def load_predictor(path: str, checkpoint_mtime: float) -> EEGPredictor:
    from inference.eeg_predictor import EEGPredictor

    del checkpoint_mtime
    return EEGPredictor(path)


def active_predictor() -> EEGPredictor | None:
    if not CHECKPOINT.is_file():
        st.warning(
            "The trained model file is not available. Prediction remains "
            "disabled until the real cohort has been processed and trained."
        )
        return None
    try:
        return load_predictor(str(CHECKPOINT), CHECKPOINT.stat().st_mtime)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        st.error(f"The trained checkpoint could not be loaded: {exc}")
        return None


def prediction_result() -> dict | None:
    result = st.session_state.get("last_eeg_prediction")
    if not result:
        st.info("Run a prediction first to view input-specific explainability.")
        return None
    return result


def show_prediction(result: dict) -> None:
    left, right = st.columns([1, 1.3])
    with left:
        predicted_label = CLASS_LABELS.get(
            result["predicted_class"], result["predicted_class"]
        )
        st.metric("Closest study-group match", predicted_label)
        st.caption(f"The model checked {result['window_count']} usable 5-second EEG segments.")
    with right:
        probabilities = result["class_probabilities"]
        probability_figure = go.Figure(
            go.Bar(
                x=[CLASS_LABELS.get(name, name) for name in probabilities],
                y=list(probabilities.values()),
                marker_color=["#b04b56", "#d59643", "#31818b"],
                text=[f"{value:.1%}" for value in probabilities.values()],
                textposition="outside",
            )
        )
        probability_figure.update_layout(
            title="Model scores for each study group",
            yaxis={"title": "Model score", "range": [0, 1.15]},
            template="plotly_white",
            height=360,
        )
        st.plotly_chart(probability_figure, width="stretch")
    st.caption(
        "These scores describe the model's output for this recording. They are not "
        "diagnoses or medical risk estimates."
    )
    with st.expander("Recording and processing details"):
        st.write(
            f"Original recording rate: "
            f"{result['preprocessing']['source_sampling_frequency_hz']:g} samples/second. "
            f"Usable segments: {result['window_count']}; rejected segments: "
            f"{result['preprocessing']['rejected_windows']}."
        )
        st.caption(result["explainability_note"])


if page == "Home":
    cohort = load_cohort()
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Project overview")
        st.write(
            "This research tool looks at an EEG recording, with optional age, sex, "
            "and memory-test information, and compares it with three study groups: "
            "Alzheimer's disease, frontotemporal dementia, and healthy controls."
        )
        st.write(
            "To try it: open **New prediction**, choose a compatible EEG file, add "
            "any known optional details, then select **Analyze recording**."
        )
    with right:
        st.subheader("System status")
        st.metric("Trained model", "Ready" if CHECKPOINT.is_file() else "Not trained")
        st.metric("Model results", "Available" if METRICS_PATH.is_file() else "Not run")
        st.caption("Files stay on this computer. A graphics card is not required.")
    if cohort:
        frame, inventory = cohort
        cols = st.columns(4)
        cols[0].metric("People in study", inventory["participants"])
        cols[1].metric("Alzheimer's disease", inventory["class_counts"]["AD"])
        cols[2].metric("Frontotemporal dementia", inventory["class_counts"]["FTD"])
        cols[3].metric("Healthy controls", inventory["class_counts"]["CN"])
        st.caption(
            f"Public CC0 data · {inventory['raw_eeg_recordings']} raw recordings · "
            f"{inventory['raw_eeg_bytes'] / (1024**3):.2f} GiB EEG"
        )
        class_counts = (
            frame["diagnosis"]
            .value_counts()
            .reindex(CLASS_NAMES, fill_value=0)
            .rename(index=CLASS_LABELS)
        )
        chart = go.Figure(
            go.Bar(
                x=class_counts.index,
                y=class_counts.values,
                marker_color=["#b04b56", "#d59643", "#31818b"],
            )
        )
        chart.update_layout(
            title="People in each study group",
            xaxis_title="Study group",
            yaxis_title="Participants",
            template="plotly_white",
        )
        st.plotly_chart(chart, width="stretch")
    metrics = load_metrics()
    if metrics:
        st.subheader("Results for people not used to train the model")
        model_results = metrics["evaluation"]
        model_labels = {
            "multimodal": "EEG + optional details",
            "eeg_only": "EEG only",
            "clinical_only": "Age, sex, and MMSE only",
        }
        table = pd.DataFrame(
            {
                model_labels.get(name, name): {
                    "Overall correct": values["accuracy"],
                    "Group-balanced score": values["balanced_accuracy"],
                    "Average F1 score": values["f1_macro"],
                    "Group-separation score": values["auroc_ovr_macro"],
                }
                for name, values in model_results.items()
            }
        ).T
        st.dataframe(table.style.format("{:.3f}", na_rep="n/a"), width="stretch")
        st.caption(
            f"Number of people checked: {metrics['split']['participant_counts']['test']}. "
            "Scores summarize a small held-out group and may vary with new data."
        )
        with st.expander("What do these scores mean?"):
            st.write(
                "**Overall correct** is the share of people assigned to the right group. "
                "**Group-balanced score** gives each group equal weight. **Average F1** "
                "balances missed and incorrect group assignments. **Group-separation "
                "score** measures how well the scores rank members of one group above "
                "others; 1 is best and 0.5 is chance."
            )
    else:
        st.info("Model results will appear after running `python evaluate.py`.")


elif page == "Dataset":
    st.subheader("People and recordings in the study")
    cohort = load_cohort()
    if cohort:
        frame, inventory = cohort
        st.caption(
            f"Source: {inventory['source']} · License: {inventory['license']} · "
            f"n={inventory['participants']}"
        )
        class_filter = st.multiselect(
            "Show study groups",
            [CLASS_LABELS[name] for name in CLASS_NAMES],
            default=[CLASS_LABELS[name] for name in CLASS_NAMES],
        )
        selected_codes = [
            code for code, label in CLASS_LABELS.items() if label in class_filter
        ]
        filtered = frame[frame["diagnosis"].isin(selected_codes)]
        display = filtered[
            [
                "participant_id",
                "diagnosis",
                "age",
                "gender",
                "mmse",
                "recording_duration_seconds",
                "sampling_frequency_hz",
                "channel_count",
            ]
        ].rename(
            columns={
                "participant_id": "Participant",
                "diagnosis": "Study group",
                "age": "Age (years)",
                "gender": "Sex",
                "mmse": "Memory-test score (MMSE)",
                "recording_duration_seconds": "Recording length (seconds)",
                "sampling_frequency_hz": "Recording rate (samples/second)",
                "channel_count": "Signal channels",
            }
        )
        display["Study group"] = display["Study group"].map(CLASS_LABELS)
        st.dataframe(display, width="stretch", hide_index=True)
        missing = inventory["missing_values"]
        st.subheader("Information availability")
        st.write(
            f"Brain-wave (EEG) recordings: {inventory['modalities']['eeg']['present']} available. "
            f"Age not recorded: {missing['age']}; sex not recorded: {missing['gender']}; "
            f"memory-test score not recorded: {missing['mmse']}."
        )
        st.caption(
            "The memory-test score (MMSE) is closely related to the study-group labels. "
            "This may make it easier for a model to rely on that score instead of EEG."
        )


elif page == "View EEG":
    st.subheader("Look at a recorded EEG signal")
    st.caption(
        "Choose a study participant to view a short, cleaned segment of their recording. "
        "The vertical offsets separate the lines for display; they are not extra signal."
    )
    cohort = load_cohort()
    if cohort:
        frame, _ = cohort
        available = [
            participant_id
            for participant_id in frame["participant_id"].astype(str)
            if (CACHE_ROOT / f"{participant_id}.npz").is_file()
        ]
        if not available:
            st.warning("Preprocessed caches are not available. Run `python preprocess.py`.")
        else:
            participant_id = st.selectbox("Study participant", available)
            cache_path = CACHE_ROOT / f"{participant_id}.npz"
            try:
                with np.load(cache_path, allow_pickle=False) as archive:
                    windows = np.asarray(archive["windows_uv"], dtype=np.float32)
                    metadata = json.loads(str(archive["metadata"].item()))
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                st.error(f"Could not read cached EEG windows: {exc}")
                windows = np.empty((0, len(CHANNELS), 0), dtype=np.float32)
                metadata = {}
            if len(windows):
                window_index = st.slider(
                    "Five-second window",
                    min_value=0,
                    max_value=len(windows) - 1,
                    value=0,
                )
                selected_channels = st.multiselect(
                    "Channels",
                    list(CHANNELS),
                    default=list(CHANNELS[:8]),
                )
                fig = go.Figure()
                time = np.arange(windows.shape[-1]) / 128.0
                offset = 0.0
                for channel in selected_channels:
                    index = CHANNELS.index(channel)
                    signal = windows[window_index, index]
                    fig.add_trace(
                        go.Scatter(
                            x=time,
                            y=signal + offset,
                            mode="lines",
                            name=channel,
                            line={"width": 1},
                        )
                    )
                    offset += max(float(np.ptp(signal)) * 1.5, 80.0)
                fig.update_layout(
                    title=f"{participant_id} · filtered EEG in microvolts",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Signal + display offset (microvolts)",
                    template="plotly_white",
                    height=620,
                    legend={"orientation": "h"},
                )
                st.plotly_chart(fig, width="stretch")
                with st.expander("Recording details"):
                    st.json(metadata, expanded=False)
                st.caption("Only segments that passed the signal-quality checks are shown.")


elif page == "New prediction":
    st.subheader("Check a new EEG recording")
    st.write(
        "Choose a standalone EEGLAB `.set` file containing the study's 19 brain-wave (EEG) "
        "channels. It is processed on this computer and the temporary upload is "
        "removed when the check finishes."
    )
    st.info(
        "Age and MMSE are optional. Allowed ranges: age 18–120 years; MMSE 0–30 "
        "whole-number points. Leave a value blank if it is unknown."
    )
    predictor = active_predictor()
    with st.form("eeg_prediction_form"):
        research_code = st.text_input(
            "Optional study code",
            help="Use a code, not a person's name or hospital number.",
        )
        eeg_file = st.file_uploader(
            "Select EEG recording (.set)",
            type=["set"],
            help=(
                "The file must include the 19 expected study channels and be no larger "
                "than 512 MiB. Separate .fdt companion files are not supported."
            ),
        )
        age = st.number_input(
            "Age in years",
            min_value=18,
            max_value=120,
            value=None,
            step=1,
            help="Enter a whole number from 18 to 120, or leave blank if unknown.",
        )
        gender = st.selectbox(
            "Sex recorded for this person",
            ["Not provided", "Female", "Male", "Other / not listed"],
            help=(
                "The study model was trained with Female and Male entries. "
                "Other / not listed is treated as not provided."
            ),
        )
        mmse = st.number_input(
            "Memory-test score (MMSE)",
            min_value=0,
            max_value=30,
            value=None,
            step=1,
            help="Enter a whole-number score from 0 to 30, or leave blank if unknown.",
        )
        submitted = st.form_submit_button(
            "Analyze recording",
            type="primary",
            disabled=predictor is None,
        )
    if submitted:
        if eeg_file is None:
            st.error("Choose an EEG recording before running inference.")
        elif eeg_file.size > 512 * 1024 * 1024:
            st.error("The uploaded EEG file exceeds the 512 MiB local limit.")
        elif predictor is None:
            st.error("The trained model could not be loaded. Check the model file and try again.")
        else:
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".set", delete=False) as temp:
                    temp.write(eeg_file.getbuffer())
                    temp_path = Path(temp.name)
                clinical = {
                    "age": age,
                    "gender": (
                        None
                        if gender in {"Not provided", "Other / not listed"}
                        else gender
                    ),
                    "mmse": mmse,
                }
                with st.spinner("Checking signal quality and analyzing the recording..."):
                    result = predictor.predict(temp_path, clinical)
                result["uploaded_file_sha256"] = hashlib.sha256(
                    eeg_file.getbuffer()
                ).hexdigest()
                result["research_code"] = research_code.strip()
                st.session_state["last_eeg_prediction"] = result
                st.success("Analysis complete.")
            except (ValueError, OSError, RuntimeError) as exc:
                st.error(f"Prediction failed: {exc}")
            finally:
                if temp_path and temp_path.exists():
                    temp_path.unlink()
    saved = st.session_state.get("last_eeg_prediction")
    if saved:
        show_prediction(saved)
        safe_code = re.sub(r"[^A-Za-z0-9._-]+", "_", saved.get("research_code", ""))
        report_name = f"neurofusion_{safe_code or 'eeg'}_report.pdf"
        st.download_button(
            "Download prediction report (PDF)",
            data=build_eeg_report(saved, research_code=saved.get("research_code", "")),
            file_name=report_name,
            mime="application/pdf",
        )


elif page == "Why this result?":
    st.subheader("What the model responded to")
    result = prediction_result()
    if result:
        st.caption(
            "These charts show patterns the model used for this recording. They do "
            "not prove why a person has a condition or identify a cause."
        )
        left, right = st.columns(2)
        with left:
            attention = result["attention_weights"]
            figure = go.Figure(
                go.Heatmap(
                    z=attention["matrix"],
                    x=attention["key_modalities"],
                    y=attention["query_modalities"],
                    text=np.asarray(attention["matrix"]).round(3),
                    texttemplate="%{text}",
                    colorscale="Teal",
                    zmin=0,
                    zmax=1,
                )
            )
            figure.update_layout(
                title="How much EEG and optional details were weighted",
                xaxis_title="Information considered",
                yaxis_title="Information being updated",
                yaxis={"autorange": "reversed"},
                template="plotly_white",
            )
            st.plotly_chart(figure, width="stretch")
        with right:
            importance = result["channel_importance"]
            ordered = sorted(importance.items(), key=lambda item: item[1])
            figure = go.Figure(
                go.Bar(
                    x=[value for _, value in ordered],
                    y=[name for name, _ in ordered],
                    orientation="h",
                    marker_color="#176b87",
                )
            )
            figure.update_layout(
                title="EEG channels that most affected this model output",
                xaxis_title="Relative effect on model output",
                yaxis_title="EEG channel",
                template="plotly_white",
            )
            st.plotly_chart(figure, width="stretch")
        st.subheader("What changes when information is hidden?")
        st.caption(
            "The model is run again with one type of information hidden. Changes are "
            "input-specific diagnostics, not proof of cause."
        )
        ablation = result["modality_ablation"]
        class_order = ablation["class_order"]
        figure = go.Figure()
        for key, name in (
            ("full_probabilities", "Full input"),
            ("without_eeg_probabilities", "Without EEG"),
            ("without_clinical_probabilities", "Without clinical"),
        ):
            figure.add_trace(
                go.Bar(
                    name=name,
                    x=[CLASS_LABELS.get(name, name) for name in class_order],
                    y=ablation[key],
                )
            )
        figure.update_layout(
            title="Model scores with different information hidden",
            yaxis={"title": "Model score", "range": [0, 1]},
            barmode="group",
            template="plotly_white",
        )
        st.plotly_chart(figure, width="stretch")


elif page == "Model results":
    st.subheader("How the models performed on held-out people")
    metrics = load_metrics()
    if not metrics:
        st.info("Run `python evaluate.py` after training to create actual held-out metrics.")
    else:
        st.caption(
            f"Test set n={metrics['split']['participant_counts']['test']} · "
            f"seed={metrics['split']['seed']} · participant windows aggregated by mean probability."
        )
        tabs = st.tabs(
            ["Compare models", "Correct / incorrect", "Group ranking", "Misses / hits", "Confidence"]
        )
        model_names = {
            "multimodal": "EEG + optional details",
            "eeg_only": "EEG only",
            "clinical_only": "Age, sex, and MMSE only",
        }
        with tabs[0]:
            rows = []
            for mode, name in model_names.items():
                values = metrics["evaluation"][mode]
                ci = values["stratified_bootstrap_95_ci"]
                rows.append(
                    {
                        "Model": name,
                        "People assessed": values["participant_count"],
                        "Overall correct": values["accuracy"],
                        "Group-balanced score": values["balanced_accuracy"],
                        "Average match accuracy": values["precision_macro"],
                        "Average group found": values["recall_macro"],
                        "Combined match score": values["f1_macro"],
                        "Group-ranking score": values["auroc_ovr_macro"],
                        "Precision/recall score": values["auprc_macro"],
                        "Probability error": values["calibration"]["brier_score"],
                        "Confidence gap": values["calibration"]["expected_calibration_error_10_bins"],
                        "F1 range (95% interval)": ci["macro_f1_95_ci"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption(
                "Each person counts once. Higher is better for the match and ranking "
                "scores; lower is better for probability error and confidence gap."
            )
            with st.expander("How to read the scores and charts"):
                st.write(
                    "**Overall correct** is the share of people assigned to the right "
                    "group. **Group-balanced score** gives each group equal weight. "
                    "**Average group found** measures how many people in each group were "
                    "found. **Correct / incorrect** compares the recorded group with the "
                    "model's choice. **Group ranking** shows how well scores separate one "
                    "group from others. **Confidence** compares model scores with observed "
                    f"correctness. These results come from only "
                    f"{metrics['split']['participant_counts']['test']} held-out people, "
                    "so they are uncertain. The 95% interval shows the range of F1 values "
                    "seen when resampling this small test group."
                )
        with tabs[1]:
            columns = st.columns(3)
            for column, (mode, name) in zip(columns, model_names.items()):
                confusion = metrics["evaluation"][mode]["confusion_matrix"]
                figure = go.Figure(
                    go.Heatmap(
                        z=confusion,
                        x=[CLASS_LABELS[name] for name in CLASS_NAMES],
                        y=[CLASS_LABELS[name] for name in CLASS_NAMES],
                        text=confusion,
                        texttemplate="%{text}",
                        colorscale="Blues",
                        showscale=False,
                    )
                )
                figure.update_layout(
                    title=name,
                    xaxis_title="Model choice",
                    yaxis_title="Recorded study group",
                    yaxis={"autorange": "reversed"},
                    template="plotly_white",
                )
                column.plotly_chart(figure, width="stretch")
        with tabs[2]:
            figure = go.Figure()
            for mode, name in model_names.items():
                for label, curve in metrics["evaluation"][mode]["roc_curves_ovr"].items():
                    figure.add_trace(
                        go.Scatter(
                            x=curve["false_positive_rate"],
                            y=curve["true_positive_rate"],
                            mode="lines",
                            name=(
                                f"{name} · {CLASS_LABELS.get(label, label)} "
                                f"(AUC {curve['auroc']:.2f})"
                            ),
                        )
                    )
            figure.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                    line={"dash": "dash"},
                )
            )
            figure.update_layout(
                title="How well scores separate each group",
                xaxis_title="False matches",
                yaxis_title="Correct matches",
                xaxis={"range": [0, 1]},
                yaxis={"range": [0, 1]},
                template="plotly_white",
            )
            st.plotly_chart(figure, width="stretch")
        with tabs[3]:
            figure = go.Figure()
            for mode, name in model_names.items():
                for label, curve in metrics["evaluation"][mode]["precision_recall_curves_ovr"].items():
                    figure.add_trace(
                        go.Scatter(
                            x=curve["recall"],
                            y=curve["precision"],
                            mode="lines",
                            name=(
                                f"{name} · {CLASS_LABELS.get(label, label)} "
                                f"(AP {curve['auprc']:.2f})"
                            ),
                        )
                    )
            figure.update_layout(
                title="Finding group members while limiting false matches",
                xaxis_title="Share of group members found",
                yaxis_title="Share of matches that were correct",
                xaxis={"range": [0, 1]},
                yaxis={"range": [0, 1]},
                template="plotly_white",
            )
            st.plotly_chart(figure, width="stretch")
        with tabs[4]:
            figure = go.Figure()
            for mode, name in model_names.items():
                bins = metrics["evaluation"][mode]["calibration"]["reliability_bins"]
                figure.add_trace(
                    go.Scatter(
                        x=[item["confidence"] for item in bins],
                        y=[item["accuracy"] for item in bins],
                        mode="lines+markers",
                        name=name,
                        text=[item["count"] for item in bins],
                        hovertemplate="Confidence %{x:.2f}<br>Accuracy %{y:.2f}"
                        "<br>Participants %{text}<extra></extra>",
                    )
                )
            figure.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal", line={"dash": "dash"})
            )
            figure.update_layout(
                title="Are higher model scores usually more correct?",
                xaxis_title="Average model score",
                yaxis_title="Observed share correct",
                xaxis={"range": [0, 1]},
                yaxis={"range": [0, 1]},
                template="plotly_white",
            )
            st.plotly_chart(figure, width="stretch")
        st.subheader("Limitations")
        for note in metrics["scope_and_limitations"]:
            st.warning(note)
        st.download_button(
            "Download measured evaluation JSON",
            data=json.dumps(metrics, indent=2),
            file_name="neurofusion_ds004504_evaluation.json",
            mime="application/json",
        )


elif page == "About this study":
    st.subheader("What this tool does—and does not do")
    st.write(
        "This tool compares a new EEG recording with examples from three groups in a "
        "public research dataset: Alzheimer's disease, frontotemporal dementia, and "
        "healthy controls. It sorts recordings into study groups; it does not diagnose "
        "a person or predict future disease progression."
    )
    st.markdown(
        """
        **In simple terms**
        1. The recording is checked and cleaned using the same steps used for the study data.
        2. It is split into short five-second pieces.
        3. A trained model looks for patterns in the EEG and, when provided, age, sex, and MMSE.
        4. The model combines the available information and reports its scores for each study group.
        5. The scores are averaged across the usable pieces of the recording.
        """
    )
    with st.expander("Technical details for research reviewers"):
        st.markdown(
            """
            - MNE reads raw EEGLAB `.set` recordings.
            - A fourth-order Butterworth 0.5–45 Hz band-pass is applied; signals are resampled to 128 Hz.
            - Non-overlapping five-second windows with extreme amplitude or non-finite values are rejected.
            - A 1D CNN encodes EEG; a separate MLP encodes age, sex, MMSE, and missing-value indicators.
            - Masked multi-head cross-modal attention combines available EEG and clinical representations.
            - People are split into training, validation, and test groups before their recordings are windowed.
            """
        )
    diagram = go.Figure()
    nodes = [
        (0.08, 0.72, "Raw EEG<br>19 channels"),
        (0.29, 0.72, "Filter + resample<br>5 s windows"),
        (0.51, 0.72, "Find signal<br>patterns"),
        (0.29, 0.28, "Age · sex · MMSE<br>if provided"),
        (0.65, 0.5, "Combine available<br>information"),
        (0.88, 0.5, "Study-group<br>scores"),
    ]
    edges = [(0, 1), (1, 2), (2, 4), (3, 4), (4, 5)]
    for start, end in edges:
        diagram.add_trace(
            go.Scatter(
                x=[nodes[start][0], nodes[end][0]],
                y=[nodes[start][1], nodes[end][1]],
                mode="lines",
                line={"color": "#83a4b8", "width": 2},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    diagram.add_trace(
        go.Scatter(
            x=[item[0] for item in nodes],
            y=[item[1] for item in nodes],
            mode="markers+text",
            text=[item[2] for item in nodes],
            textposition="middle center",
            marker={"size": 62, "color": "#176b87", "line": {"color": "white", "width": 2}},
            textfont={"color": "white", "size": 11},
            hoverinfo="text",
            showlegend=False,
        )
    )
    diagram.update_layout(
        title="How a recording is analyzed",
        template="plotly_white",
        height=430,
        xaxis={"visible": False, "range": [0, 1]},
        yaxis={"visible": False, "range": [0, 1]},
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    st.plotly_chart(diagram, width="stretch")
    st.markdown(
        """
        **Data and references**
        - Dataset: [NEMAR on004504](https://www.nemar.org/dataset/on004504), public CC0 copy of OpenNeuro ds004504.
        - Source dataset: 88 participants; 36 AD, 23 FTD, 29 controls; 19 scalp EEG channels; 500 Hz original sampling.
        - Dataset descriptor: Ieracitano et al., *A Dataset of Scalp EEG Recordings of Alzheimer’s Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG*, Data, 2023, 8(6), 95. DOI: 10.3390/data8060095.
        - The supplied IEEE paper is a survey of 15 studies and a proposed framework, not evidence that this project's model has clinical validity.
        """
    )
    st.warning(
        "The study includes only 88 people from one source. The MMSE score is closely "
        "related to the group labels and may dominate results. The model has not been "
        "independently validated; outputs and explanation charts are for research only."
    )
