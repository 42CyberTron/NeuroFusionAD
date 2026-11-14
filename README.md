# NeuroFusion AD

NeuroFusion AD is a local research application for **cross-sectional EEG and
clinical classification of Alzheimer's disease (AD), frontotemporal dementia
(FTD), and healthy controls (CN)**. It uses the public CC0 NEMAR copy of
OpenNeuro ds004504. It does **not** predict MCI-to-AD progression, and it is not
a medical device.

The supplied IEEE paper is a survey of 15 studies and a proposed research
direction; it does not report trained NeuroFusion AD weights or project-specific
performance. This implementation trains and evaluates its own models from the
dataset downloaded below. Reported scores are generated from held-out
participants by this code, not copied from the paper.

## Dataset inspection and provenance

Before choosing the input pipeline, the public NEMAR copy was checked directly:

- NEMAR dataset page: <https://www.nemar.org/dataset/on004504>
- Dataset metadata, BIDS sidecars, channel table, README, and `participants.tsv`
  identify 88 people: 36 AD (`A`), 23 FTD (`F`), and 29 healthy controls (`C`).
- The recordings are continuous eyes-closed resting EEG, 19 scalp channels,
  500 Hz original sampling rate; the actual participant-specific durations
  and file sizes are inventoried after download.
- Clinical fields are age, gender, and MMSE. The source codebook maps `M/F`
  to male/female; group labels map as `A -> AD`, `F -> FTD`, `C -> CN`.
- The NEMAR copy declares CC0. The acquisition script uses that public NEMAR
  endpoint and downloads only raw BIDS recordings and required metadata; it
  does not retrieve the separate preprocessed derivatives.
- The downloaded BIDS metadata identifies source OpenNeuro version 1.0.9
  (`doi:10.18112/openneuro.ds004504.v1.0.9`) and the NEMAR copy as version 1.0.0.
- The local verification found 2,829,786,488 bytes of raw EEG, duration
  307.1–1,291.1 seconds (median 814.05 seconds), and no missing EEG, age,
  gender, or MMSE values among the 88 participants.

NEMAR's dataset README describes a previously applied device/software filter
and explains that the raw files do not include the EEG device's artifact
annotations. This project therefore does not claim to reproduce the dataset's
ASR/ICA-cleaned derivatives. It applies the documented local pipeline below to
the raw `.set` files. The BIDS sidecars describe an A1/A2 reference while only
the 19 scalp channels are supplied, so no additional reference transform is
applied.

## Local setup (Windows PowerShell)

Use Python 3.11 (or a compatible Python version supported by PyTorch and MNE):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
python stress_test.py --iterations 20
```

`stress_test.py` checks model tensor shapes, finite outputs, and local forward
latency using abstract test tensors only; it is not a trained-model or clinical
performance benchmark.

GPU is optional. Training automatically selects CUDA if available and otherwise
uses CPU. To download the authentic cohort, preprocess it, train all three
comparators, and compute held-out results:

```powershell
python -m datasets.download_openneuro --output data/openneuro_ds004504 --workers 4
python -m preprocessing.process_openneuro --data data/openneuro_ds004504 --output data/processed/ds004504
python train.py
python evaluate.py
```

The download is roughly a few gigabytes of raw `.set` files. Check that the
computer has enough free disk space before starting. The download script
validates the CC0 license, BIDS sidecars, 19-channel layout, sampling rate,
group labels, and file sizes. It can be rerun to skip files already downloaded
and verified. Acquisition requires internet; preprocessing, training,
evaluation, dashboard use, and inference run locally after acquisition and
dependency installation.

Launch the dashboard:

```powershell
python -m streamlit run app.py
```

Run inference from a compatible, previously unseen EEGLAB recording:

```powershell
python predict.py --eeg "D:\path\to\new_recording.set" --age 67 --gender Female --mmse 24
```

Age, gender, and MMSE may be omitted. Missing values are imputed using
training-participant statistics and explicit missingness indicators. The model
was trained with female/male source values only; `Other` is treated as
unrepresented/missing rather than silently mapped to a trained sex category.
The uploaded EEG must contain the 19 expected channels; external `.fdt` bundles
are not accepted. The dashboard keeps the upload in a temporary local file and
removes it after inference.

The prediction form accepts age as a whole number from **18 to 120** and MMSE
as a whole-number score from **0 to 30**. Both are optional; leave either blank
when it is unknown. The sex selector offers Female, Male, Other / not listed,
and Not provided. Other / not listed is treated as unavailable because the
training data contains only Female and Male entries. EEG uploads are limited to
512 MiB.

## Preprocessing and modeling

`preprocessing/eeg.py` reads EEGLAB `.set` files with MNE and:

1. Validates and orders the 19 source channels consistently.
2. Applies a fourth-order Butterworth band-pass from 0.5 to 45 Hz.
3. Resamples to 128 Hz.
4. Segments signals into non-overlapping 5-second windows.
5. Rejects windows containing non-finite samples, any-channel peak-to-peak
   amplitude above 500 microvolts, or an absolute peak above 500 microvolts.
6. Keeps the number of rejected and accepted windows in the dataset inventory.

This is an explicit gross-amplitude artifact rule, not ASR, ICA, or expert
artifact annotation. It does not repair subtle ocular, muscle, or movement
artifacts. `data/processed/ds004504/dataset_inventory.json` records actual
sample counts, class counts, missing clinical values, signal durations,
accepted/rejected windows, and preprocessing decisions.

On the downloaded cohort, the pipeline produced 14,074 possible complete
windows; 13,550 passed the finite-signal and amplitude checks, while 524 were
rejected. These are preprocessing counts, not model-performance results.

The model has a 1D CNN EEG encoder, an MLP clinical encoder, a masked
multi-head cross-modal attention block, and three-class logits. Three models
are trained with the same participant split:

- EEG + clinical attention model.
- EEG-only baseline.
- Clinical-only baseline.

Each dataset participant is split before windowing into training, validation,
or test; no participant's windows cross partitions. Clinical and per-channel
normalization statistics are fitted on training participants only. The
training sampler balances classes and participants, and early stopping uses
participant-level validation log loss. The multimodal model includes
training-time clinical-modality dropout so it can execute when clinical inputs
are missing. The final test partition is not used for model selection.

The final evaluation averages each held-out participant's window probabilities
and reports accuracy, balanced accuracy, macro precision/recall/F1,
one-vs-rest AUROC/AUPRC where classes permit, confusion matrices, Brier score,
10-bin expected calibration error, per-class results, and participant-level
stratified bootstrap intervals. `evaluation/metrics.json` contains the measured
outputs; offline interactive diagrams are written to `evaluation/figures/`.

### Measured held-out results

The fixed seed-42 split used 56 training participants, 14 validation
participants, and 18 held-out test participants (7 AD, 5 FTD, 6 CN). Results
below are from the actual saved weights; windows were aggregated per person.

| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 (95% bootstrap CI) | Macro AUROC | Macro AUPRC | Brier | 10-bin ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EEG + clinical attention | 0.778 | 0.733 | 0.843 | 0.733 | 0.688 (0.558–0.869) | 0.924 | 0.885 | 0.284 | 0.102 |
| EEG only | 0.500 | 0.498 | 0.502 | 0.498 | 0.493 (0.264–0.717) | 0.702 | 0.558 | 0.638 | 0.282 |
| Clinical only | 0.889 | 0.867 | 0.926 | 0.867 | 0.875 (0.704–1.000) | 0.934 | 0.893 | 0.202 | 0.119 |

Confusion matrices use true/predicted class order AD, FTD, CN:

```text
EEG + clinical: [[7, 0, 0], [2, 1, 2], [0, 0, 6]]
EEG only:       [[3, 3, 1], [1, 2, 2], [1, 1, 4]]
Clinical only:  [[7, 0, 0], [2, 3, 0], [0, 0, 6]]
```

The clinical-only model outperformed the fused model on this split; the EEG-only
baseline was weaker. The fused model correctly classified only 1 of 5 held-out
FTD participants (FTD recall 0.20). These are small-sample exploratory
measurements, not a basis for a clinical claim. In particular, MMSE is strongly
associated with the source labels, and clinical-only performance likely reflects
that association. The attention model did not demonstrate an improvement over
the clinical baseline here.

Offline diagrams: [`evaluation/figures/index.html`](./evaluation/figures/index.html).
Detailed values and per-class curves: [`evaluation/metrics.json`](./evaluation/metrics.json).
Summary: [`reports/validation_findings.html`](./reports/validation_findings.html).

### Interpretation limits

- The 88-person cohort is small and single-source. One held-out split is not
  external validation, and performance uncertainty is substantial.
- MMSE is strongly associated with the source labels (the healthy group has
  MMSE 30 in the supplied metadata). The clinical-only baseline is reported to
  make this shortcut risk visible; EEG-only performance is an important
  comparator.
- Attention maps and gradient-times-input channel scores are descriptive model
  diagnostics, not causal explanations.
- A model probability is not a calibrated clinical risk or medical diagnosis.
- This dataset supports AD/FTD/CN cross-sectional classification. It does not
  support MCI-to-AD longitudinal progression claims.

## Dashboard guide

The left menu uses plain-language sections:

- **Home:** project status, study-group counts, and a brief result summary.
- **Dataset:** browse the study participants and their available information.
- **View EEG:** inspect a short, cleaned signal segment.
- **New prediction:** upload an EEG file, optionally enter age/sex/MMSE, and
  select **Analyze recording**. The report can be downloaded after analysis.
- **Why this result?:** inspect model-response charts. These are exploratory
  diagnostics, not explanations of a medical cause.
- **Model results:** compare EEG-only, clinical-only, and combined models.
- **About this study:** read the workflow, research methods, and limitations.

## Testing

```powershell
python -m pytest
```

Tests cover model attention and missing modalities, participant-level split
integrity, MNE preprocessing mechanics, input validation, metric calculations,
and the requirement for a trained inference checkpoint. Software-unit-test
signals are not used for model training or reported performance. The
full-cohort train/evaluate commands are the reproducible research run.

A separate CPU stress run made 100 forward passes at each batch size using
abstract test tensors only (400 forwards total). Mean tensor-forward latencies
were 0.916 ms (batch 1), 1.890 ms (batch 8), 4.283 ms (batch 32), and 7.659 ms
(batch 64) on the local CPU with four PyTorch threads. These do not include MNE
file parsing and are not trained-model or clinical inference benchmarks.
Details: [`evaluation/stress_test.json`](./evaluation/stress_test.json).

## Main project files

```text
datasets/download_openneuro.py   Authorized public-data acquisition
datasets/openneuro.py            BIDS and participant metadata validation
preprocessing/eeg.py             MNE filtering, resampling, windowing
preprocessing/process_openneuro.py Preprocessed caches and actual inventory
models/eeg_network.py             CNN + clinical encoder + masked attention
training/eeg_data.py              Subject splits and train-only normalization
training/eeg_train.py             Three-model training and checkpoints
inference/eeg_predictor.py        Local trained-model inference and explanations
evaluation/eeg_evaluate.py        Held-out participant evaluation
evaluation/eeg_plots.py           Offline interactive diagrams
app.py                            Streamlit research dashboard
```

