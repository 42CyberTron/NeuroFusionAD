"""Trained-checkpoint inference for unseen compatible EEGLAB recordings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from datasets.openneuro import CHANNELS, CLASS_NAMES
from models.eeg_network import EEGClinicalNet
from preprocessing.eeg import preprocess_eeg_file
from training.eeg_data import encode_clinical, standardize_windows


class EEGPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path = "checkpoints/eeg_multimodal.pt",
        *,
        device: str | None = None,
    ) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"No trained EEG checkpoint found at {path}. "
                "Download ds004504 and run the training workflow first."
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        if (
            not checkpoint.get("trained")
            or checkpoint.get("mode") != "multimodal"
            or not checkpoint.get("state_dict")
        ):
            raise ValueError("Checkpoint is not a completed multimodal EEG model.")
        if checkpoint.get("class_names") != list(CLASS_NAMES):
            raise ValueError("Checkpoint class order does not match AD/FTD/CN.")
        self.checkpoint = checkpoint
        self.config = checkpoint["config"]
        self.clinical_scaling = checkpoint["clinical_scaling"]
        self.channel_scaling = checkpoint["channel_scaling"]
        self.model = EEGClinicalNet(
            clinical_features=len(self.config["clinical_features"]),
            channels=int(self.config["channels"]),
            hidden_size=int(self.config["hidden_size"]),
            dropout=float(self.config["dropout"]),
            mode="multimodal",
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()

    def predict(
        self,
        eeg_path: str | Path,
        clinical: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clinical_vector, clinical_present = encode_clinical(
            clinical or {}, self.clinical_scaling
        )
        processed = preprocess_eeg_file(eeg_path)
        windows = standardize_windows(processed.windows_uv, self.channel_scaling)
        return self._predict_windows(
            windows,
            clinical_vector,
            clinical_present,
            preprocessing={
                "sampling_frequency_hz": int(self.config["sampling_frequency_hz"]),
                "window_seconds": int(self.config["window_seconds"]),
                "source_sampling_frequency_hz": processed.original_sfreq,
                "duration_seconds": processed.duration_seconds,
                "accepted_windows": int(len(windows)),
                "rejected_windows": processed.rejected_windows,
            },
            clinical_input=clinical or {},
        )

    def _predict_windows(
        self,
        windows: np.ndarray,
        clinical: np.ndarray,
        clinical_present: bool,
        *,
        preprocessing: dict[str, Any],
        clinical_input: dict[str, Any],
    ) -> dict[str, Any]:
        if not len(windows):
            raise ValueError("Prediction requires at least one valid EEG window.")
        probabilities: list[np.ndarray] = []
        attention: list[np.ndarray] = []
        batch_size = int(self.config["batch_size"])
        with torch.no_grad():
            for start in range(0, len(windows), batch_size):
                selected = windows[start : start + batch_size]
                eeg_tensor = torch.from_numpy(selected).to(self.device)
                clinical_tensor = torch.from_numpy(
                    np.repeat(clinical[None, :], len(selected), axis=0)
                ).to(self.device)
                mask = torch.tensor(
                    [[True, clinical_present]] * len(selected),
                    dtype=torch.bool,
                    device=self.device,
                )
                output = self.model(eeg_tensor, clinical_tensor, mask)
                probabilities.append(
                    torch.softmax(output["logits"], dim=1).cpu().numpy()
                )
                attention.append(
                    output["attention_weights"].mean(dim=1).cpu().numpy()
                )
        window_probabilities = np.concatenate(probabilities, axis=0)
        mean_probability = window_probabilities.mean(axis=0)
        predicted_index = int(mean_probability.argmax())
        attention_matrix = np.concatenate(attention, axis=0).mean(axis=0)

        channel_importance = self._channel_attribution(
            windows, clinical, clinical_present, predicted_index
        )
        ablation = self._modality_ablation(
            windows, clinical, clinical_present
        )
        return {
            "predicted_class": CLASS_NAMES[predicted_index],
            "class_probabilities": {
                name: float(mean_probability[index])
                for index, name in enumerate(CLASS_NAMES)
            },
            "window_count": int(len(windows)),
            "attention_weights": {
                "query_modalities": ["EEG", "Clinical"],
                "key_modalities": ["EEG", "Clinical"],
                "matrix": attention_matrix.tolist(),
            },
            "channel_importance": channel_importance,
            "modality_ablation": ablation,
            "preprocessing": preprocessing,
            "clinical_input": {
                "age": clinical_input.get("age"),
                "gender": clinical_input.get("gender"),
                "mmse": clinical_input.get("mmse"),
                "clinical_token_present": bool(clinical_present),
            },
            "explainability_note": (
                "Attention and gradient-times-input channel scores are descriptive "
                "model diagnostics, not causal or clinically validated explanations. "
                "Ablation deltas measure sensitivity for this input only."
            ),
            "device": str(self.device),
        }

    def _channel_attribution(
        self,
        windows: np.ndarray,
        clinical: np.ndarray,
        clinical_present: bool,
        target: int,
    ) -> dict[str, float]:
        max_windows = 32
        indices = np.linspace(
            0, len(windows) - 1, min(len(windows), max_windows), dtype=int
        )
        eeg_tensor = torch.from_numpy(windows[indices]).to(self.device)
        eeg_tensor.requires_grad_(True)
        clinical_tensor = torch.from_numpy(
            np.repeat(clinical[None, :], len(indices), axis=0)
        ).to(self.device)
        mask = torch.tensor(
            [[True, clinical_present]] * len(indices),
            dtype=torch.bool,
            device=self.device,
        )
        with torch.enable_grad():
            output = self.model(eeg_tensor, clinical_tensor, mask)
            gradient = torch.autograd.grad(
                output["logits"][:, target].sum(), eeg_tensor, retain_graph=False
            )[0]
            attribution = (gradient * eeg_tensor).abs().mean(dim=(0, 2))
        values = attribution.detach().cpu().numpy()
        return {
            name: float(values[index])
            for index, name in sorted(
                enumerate(CHANNELS), key=lambda item: values[item[0]], reverse=True
            )
        }

    def _modality_ablation(
        self,
        windows: np.ndarray,
        clinical: np.ndarray,
        clinical_present: bool,
    ) -> dict[str, Any]:
        full: list[np.ndarray] = []
        no_eeg: list[np.ndarray] = []
        no_clinical: list[np.ndarray] = []
        batch_size = int(self.config["batch_size"])
        with torch.no_grad():
            for start in range(0, len(windows), batch_size):
                selected = windows[start : start + batch_size]
                count = len(selected)
                eeg_tensor = torch.from_numpy(selected).to(self.device)
                clinical_tensor = torch.from_numpy(
                    np.repeat(clinical[None, :], count, axis=0)
                ).to(self.device)
                mask = torch.tensor(
                    [[True, clinical_present]] * count,
                    dtype=torch.bool,
                    device=self.device,
                )
                full.append(
                    torch.softmax(
                        self.model(eeg_tensor, clinical_tensor, mask)["logits"], dim=1
                    ).cpu().numpy()
                )
                without_eeg = mask.clone()
                without_eeg[:, 0] = False
                no_eeg.append(
                    torch.softmax(
                        self.model(eeg_tensor, clinical_tensor, without_eeg)["logits"], dim=1
                    ).cpu().numpy()
                )
                without_clinical = mask.clone()
                without_clinical[:, 1] = False
                no_clinical.append(
                    torch.softmax(
                        self.model(eeg_tensor, clinical_tensor, without_clinical)["logits"],
                        dim=1,
                    ).cpu().numpy()
                )
        full_probability = np.concatenate(full, axis=0).mean(axis=0)
        eeg_ablated = np.concatenate(no_eeg, axis=0).mean(axis=0)
        clinical_ablated = np.concatenate(no_clinical, axis=0).mean(axis=0)
        return {
            "full_probabilities": full_probability.tolist(),
            "without_eeg_probabilities": eeg_ablated.tolist(),
            "without_clinical_probabilities": clinical_ablated.tolist(),
            "class_order": list(CLASS_NAMES),
        }
