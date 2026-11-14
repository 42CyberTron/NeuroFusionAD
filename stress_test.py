"""Software-only inference throughput check; it is not an EEG/model benchmark."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from models.eeg_network import EEGClinicalNet


def run_stress(iterations: int = 20) -> dict:
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = EEGClinicalNet(dropout=0.0, mode="multimodal").to(device).eval()
    results = []

    for batch_size in (1, 8, 32, 64):
        # Abstract software-test tensors only; not patient EEG or training data.
        eeg = torch.randn(batch_size, 19, 640, device=device)
        clinical = torch.randn(batch_size, 7, device=device)
        modality_mask = torch.ones(batch_size, 2, dtype=torch.bool, device=device)
        modality_mask[::2, 1] = False

        with torch.inference_mode():
            for _ in range(2):
                model(eeg, clinical, modality_mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            elapsed_ms = []
            for _ in range(iterations):
                started = time.perf_counter()
                output = model(eeg, clinical, modality_mask)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms.append((time.perf_counter() - started) * 1000)
                if output["logits"].shape != (batch_size, 3):
                    raise RuntimeError("Classifier returned an unexpected output shape.")
                if output["attention_weights"].shape != (batch_size, 4, 2, 2):
                    raise RuntimeError("Attention returned an unexpected shape.")
                if not all(torch.isfinite(tensor).all() for tensor in output.values()):
                    raise RuntimeError("Model produced a non-finite tensor.")
        mean_ms = statistics.mean(elapsed_ms)
        results.append(
            {
                "batch_size": batch_size,
                "windows_per_inference": batch_size,
                "iterations": iterations,
                "mean_forward_ms": round(mean_ms, 3),
                "p95_forward_ms": round(float(np.percentile(elapsed_ms, 95)), 3),
                "abstract_windows_per_second": round(batch_size * 1000 / mean_ms, 2),
                "all_outputs_finite": True,
            }
        )
    return {
        "scope": (
            "Randomized abstract tensors used only for software stress testing. "
            "No dataset, trained checkpoint, diagnostic output, or model performance "
            "is used or measured."
        ),
        "torch_version": torch.__version__,
        "device": str(device),
        "threads": torch.get_num_threads(),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", help="Optional JSON results path.")
    args = parser.parse_args()
    results = run_stress(args.iterations)
    rendered = json.dumps(results, indent=2)
    print(rendered)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
