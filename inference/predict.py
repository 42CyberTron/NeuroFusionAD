"""CLI prediction for a JSON file containing genuine patient visits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference.predictor import NeuroFusionPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/neurofusion.pt")
    parser.add_argument("--input", required=True, help="JSON object with a 'visits' list")
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = NeuroFusionPredictor(args.checkpoint).predict(payload["visits"])
    result["mri_gradcam"] = None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
