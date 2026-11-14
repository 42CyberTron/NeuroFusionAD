"""Run inference on a previously unseen compatible EEGLAB .set recording."""
from __future__ import annotations

import argparse
import json

from inference.eeg_predictor import EEGPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eeg", required=True, help="Local EEGLAB .set recording.")
    parser.add_argument(
        "--age", type=float, default=None, help="Age in years (18-120)."
    )
    parser.add_argument(
        "--gender",
        choices=("Male", "Female", "Other", "Not recorded"),
        default="Not recorded",
    )
    parser.add_argument(
        "--mmse", type=float, default=None, help="Memory-test score from 0 to 30."
    )
    parser.add_argument("--checkpoint", default="checkpoints/eeg_multimodal.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    clinical = {
        "age": args.age,
        "gender": None if args.gender == "Not recorded" else args.gender,
        "mmse": args.mmse,
    }
    result = EEGPredictor(args.checkpoint, device=args.device).predict(args.eeg, clinical)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
