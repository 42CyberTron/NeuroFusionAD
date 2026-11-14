from reports.pdf import build_patient_report


def test_report_generator_returns_pdf_bytes():
    patient = {"patient_id": "unit-test", "display_name": "Unit Test"}
    visits = [{"visit_date": "2025-01-01", "clinical": {"mmse": 24.0}}]
    prediction = {
        "diagnosis": {"CN": 0.1, "MCI": 0.8, "AD": 0.1},
        "progression": {"6_month": 0.2},
        "modality_ablation_delta": {"clinical": 0.0, "mri": 0.0, "pet": 0.0},
        "clinical_feature_gradient_attribution": {"mmse": 0.2},
        "mri_gradcam": None,
    }
    data = build_patient_report(patient, visits, prediction)
    assert data.startswith(b"%PDF")
