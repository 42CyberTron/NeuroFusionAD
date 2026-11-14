"""Create a factual inventory from an authorized local manifest."""
from __future__ import annotations

import argparse

from datasets.manifest import load_manifest, write_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="checkpoints/dataset_inventory.json")
    args = parser.parse_args()
    frame = load_manifest(args.manifest)
    destination = write_inventory(frame, args.output)
    print(f"Inventory written to {destination.resolve()}")


if __name__ == "__main__":
    main()
