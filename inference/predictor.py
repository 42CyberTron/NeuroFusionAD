"""Load a trained checkpoint and infer on new patient visits."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from captum.attr import IntegratedGradients

from models.network import NeuroFusionNet
from explainability.gradcam import normalize_gradcam
from preprocessing.clinical import canonicalize_clinical
from preprocessing.imaging import load_nifti_slice


class NeuroFusionPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"No trained checkpoint found at {path}. Train on an authorized cohort first."
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        if not checkpoint.get("trained") or not checkpoint.get("state_dict"):
            raise ValueError("Checkpoint is not marked as a completed trained model.")
        self.checkpoint = checkpoint
        self.config = checkpoint["config"]
        self.features = list(self.config["clinical_features"])
        if "sex" in self.features:
            raise ValueError(
                "Checkpoint uses scalar sex input; retrain with categorical sex features."
            )
        self.scaling = checkpoint["clinical_scaling"]
        self.horizons = [int(value) for value in self.config["horizons_months"]]
        self.diagnosis_labels = list(self.config["diagnosis_labels"])
        self.image_size = int(self.config["image_size"])
        self.max_visits = int(self.config["max_visits"])
        self.model = NeuroFusionNet(
            clinical_features=len(self.features),
            hidden_size=int(self.config["hidden_size"]),
            num_diagnoses=len(self.diagnosis_labels),
            num_horizons=len(self.horizons),
            dropout=float(self.config["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()

    def _prepare(self, visits: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not visits:
            raise ValueError("At least one visit is required.")
        if len(visits) > self.max_visits:
            visits = visits[-self.max_visits :]
        visits = sorted(visits, key=lambda visit: str(visit["visit_date"]))
        count = len(visits)
        size = self.image_size
        clinical = np.zeros((1, count, len(self.features)), dtype=np.float32)
        mri = torch.zeros((1, count, 1, size, size), dtype=torch.float32)
        pet = torch.zeros_like(mri)
        modality_mask = torch.zeros((1, count, 3), dtype=torch.bool)
        intervals = np.zeros((1, count), dtype=np.float32)
        parsed_dates: list[datetime] = []
        for item in visits:
            try:
                parsed_dates.append(datetime.fromisoformat(str(item["visit_date"])[:10]))
            except (KeyError, ValueError) as exc:
                raise ValueError("Each visit requires a valid YYYY-MM-DD visit_date.") from exc

        for visit_idx, item in enumerate(visits):
            clinical_values = item.get("clinical", {})
            if not isinstance(clinical_values, dict):
                raise ValueError("Visit clinical values must be a mapping.")
            clinical_values = canonicalize_clinical(clinical_values)
            has_clinical = False
            for feature_idx, feature in enumerate(self.features):
                value = clinical_values.get(feature)
                if value is not None and value != "":
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"{feature} must be a numeric value.") from exc
                    if not np.isfinite(numeric):
                        raise ValueError(f"{feature} must be finite.")
                    has_clinical = True
                else:
                    numeric = self.scaling[feature]["median"]
                stats = self.scaling[feature]
                clinical[0, visit_idx, feature_idx] = (
                    numeric - stats["median"]
                ) / stats["scale"]
            modality_mask[0, visit_idx, 0] = has_clinical
            for column, modality_idx, target in (
                ("mri_path", 1, mri),
                ("pet_path", 2, pet),
            ):
                path = item.get(column)
                if path:
                    tensor, _ = load_nifti_slice(path, size)
                    target[0, visit_idx] = tensor
                    modality_mask[0, visit_idx, modality_idx] = True
            intervals[0, visit_idx] = (
                parsed_dates[visit_idx] - parsed_dates[0]
            ).total_seconds() / 86400.0
        if not modality_mask.any():
            raise ValueError("At least one clinical value or image is required.")
        return {
            "clinical": torch.from_numpy(clinical).to(self.device),
            "mri": mri.to(self.device),
            "pet": pet.to(self.device),
            "modality_mask": modality_mask.to(self.device),
            "intervals_days": torch.from_numpy(intervals).to(self.device),
            "visit_mask": torch.ones((1, count), dtype=torch.bool, device=self.device),
        }

    @torch.enable_grad()
    def predict(self, visits: list[dict[str, Any]]) -> dict[str, Any]:
        inputs = self._prepare(visits)
        clinical = inputs["clinical"].detach().requires_grad_(True)
        mri = inputs["mri"].detach().requires_grad_(True)
        pet = inputs["pet"]
        model_inputs = {**inputs, "clinical": clinical, "mri": mri}
        activations: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}

        def capture(_module: torch.nn.Module, _args: tuple, output: torch.Tensor) -> None:
            activations["mri"] = output
            output.register_hook(lambda grad: gradients.__setitem__("mri", grad))

        handle = self.model.mri_encoder.features[9].register_forward_hook(capture)
        try:
            output = self.model(**model_inputs)
            probabilities = torch.softmax(output["diagnosis_logits"], dim=-1)[0]
            progression = torch.sigmoid(output["progression_logits"])[0]
            target_index = int(progression.argmax().item())
            score = output["progression_logits"][0, target_index]
            self.model.zero_grad(set_to_none=True)
            score.backward()
        finally:
            handle.remove()

        with torch.no_grad():
            ablation: dict[str, float] = {}
            for name, modality_index in (("clinical", 0), ("mri", 1), ("pet", 2)):
                altered = dict(inputs)
                changed_mask = inputs["modality_mask"].clone()
                changed_mask[:, :, modality_index] = False
                altered["modality_mask"] = changed_mask
                if name == "clinical":
                    altered["clinical"] = torch.zeros_like(inputs["clinical"])
                elif name == "mri":
                    altered["mri"] = torch.zeros_like(inputs["mri"])
                else:
                    altered["pet"] = torch.zeros_like(inputs["pet"])
                ablated = self.model(**altered)
                probability = torch.sigmoid(ablated["progression_logits"][0, target_index])
                ablation[name] = float(progression[target_index] - probability)

        gradcam = None
        if "mri" in activations and "mri" in gradients and inputs["modality_mask"][0, :, 1].any():
            activation = activations["mri"]
            gradient = gradients["mri"]
            latest_mri = int(torch.where(inputs["modality_mask"][0, :, 1])[0][-1])
            present_visits = inputs["modality_mask"][0, :, 1].nonzero().flatten()
            flat_index = int((present_visits == latest_mri).nonzero().flatten()[0])
            gradcam = normalize_gradcam(
                activation, gradient, self.image_size, flat_index
            )

        def clinical_score(clinical_batch: torch.Tensor) -> torch.Tensor:
            batch_size = clinical_batch.shape[0]
            expanded = {
                key: value.expand(batch_size, *value.shape[1:])
                for key, value in inputs.items()
            }
            expanded["clinical"] = clinical_batch
            return self.model(**expanded)["progression_logits"][:, target_index]

        attribution = IntegratedGradients(clinical_score).attribute(
            clinical,
            baselines=torch.zeros_like(clinical),
            n_steps=16,
        )
        feature_attribution = (
            (attribution.abs() * inputs["visit_mask"].unsqueeze(-1))
            .mean(dim=1)[0]
            .detach()
            .cpu()
            .numpy()
        )
        return {
            "diagnosis": {
                label: float(probabilities[idx].item())
                for idx, label in enumerate(self.diagnosis_labels)
            },
            "progression": {
                f"{horizon}_month": float(progression[idx].item())
                for idx, horizon in enumerate(self.horizons)
            },
            "modality_ablation_delta": ablation,
            "clinical_feature_gradient_attribution": {
                feature: float(feature_attribution[idx])
                for idx, feature in enumerate(self.features)
            },
            "mri_gradcam": gradcam,
            "trained_checkpoint": str(self.checkpoint.get("dataset_manifest", "unknown")),
            "device": str(self.device),
        }
