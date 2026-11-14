from reports.eeg_pdf import build_eeg_report


def test_eeg_report_is_a_pdf_with_research_limitations():
    result = {
        "predicted_class": "AD",
        "class_probabilities": {"AD": 0.6, "FTD": 0.2, "CN": 0.2},
        "window_count": 10,
        "device": "cpu",
        "clinical_input": {"age": 65, "gender": "Female", "mmse": 24},
        "preprocessing": {
            "source_sampling_frequency_hz": 500,
            "sampling_frequency_hz": 128,
            "rejected_windows": 1,
        },
        "attention_weights": {"matrix": [[0.6, 0.4], [0.3, 0.7]]},
        "channel_importance": {"Fp1": 0.2, "Fz": 0.1},
    }
    report = build_eeg_report(result, research_code="test-01")
    assert report.startswith(b"%PDF")
    assert len(report) > 500
