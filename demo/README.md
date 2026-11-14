# NeuroFusion AD local run

The Windows launchers start the real Streamlit EEG research dashboard. They do
not include mock patient records, synthetic EEG training data, or predetermined
predictions.

## First-time setup

From the project root, follow the dataset and training commands in
[`README.md`](../README.md):

1. Install the project requirements.
2. Download the public CC0 ds004504 raw EEG data from NEMAR.
3. Run preprocessing, train the three models, and evaluate the held-out
   participants.
4. Start the dashboard.

## Launch

Double-click `run_demo.bat`, or run from PowerShell in the project root:

```powershell
.\demo\run_demo.ps1
```

The application opens at `http://127.0.0.1:8501`. Without
`checkpoints\eeg_multimodal.pt`, the dashboard remains available but clearly
disables prediction. Once a real model checkpoint exists, use a compatible
19-channel EEGLAB `.set` recording for inference.

Predictions are research classifications, not diagnoses or clinical risks.
