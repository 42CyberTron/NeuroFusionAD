"""Download only the raw BIDS EEG recordings and metadata from NEMAR's CC0 copy."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from datasets.openneuro import NEMAR_DATASET_ID, NEMAR_VERSION, load_dataset_index


BASE_URL = f"https://data.nemar.org/{NEMAR_DATASET_ID}/{NEMAR_VERSION}"
METADATA_FILES = (
    "README.md",
    "CHANGES",
    "dataset_description.json",
    "participants.json",
    "participants.tsv",
)


def _get_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=(15, 60))
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError(f"Unexpected directory response from {url}")
    return result


def _download(url: str, destination: Path, expected_size: int | None = None) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and (
        expected_size is None or destination.stat().st_size == expected_size
    ):
        return destination.stat().st_size

    partial = destination.with_name(destination.name + ".partial")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with requests.get(
                url,
                stream=True,
                timeout=(20, 120),
                headers={"Accept-Encoding": "identity"},
            ) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            actual_size = partial.stat().st_size
            if expected_size is not None and actual_size != expected_size:
                raise IOError(
                    f"Expected {expected_size} bytes for {destination.name}, "
                    f"received {actual_size}."
                )
            os.replace(partial, destination)
            return actual_size
        except Exception as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"Could not download {url}") from last_error


def download_dataset(output: str | Path, workers: int = 4) -> Path:
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in METADATA_FILES:
        _download(f"{BASE_URL}/{quote(name)}", root / name)

    participants_path = root / "participants.tsv"
    with participants_path.open("r", encoding="utf-8-sig", newline="") as stream:
        participants = list(csv.DictReader(stream, delimiter="\t"))
    if len(participants) != 88:
        raise ValueError(f"Expected 88 participants in source metadata; got {len(participants)}.")

    downloads: list[tuple[str, Path, int | None]] = []
    for row in participants:
        participant_id = row.get("participant_id", "").strip()
        if not participant_id.startswith("sub-"):
            raise ValueError(f"Invalid participant identifier: {participant_id!r}")
        listing = _get_json(
            f"{BASE_URL}/{participant_id}/eeg/?format=json"
        )
        children = listing.get("children")
        if not isinstance(children, list):
            raise ValueError(f"Missing file listing for {participant_id}.")
        wanted = [
            child
            for child in children
            if child.get("kind") == "file"
            and (
                str(child.get("name", "")).endswith("_eeg.set")
                or str(child.get("name", "")).endswith("_eeg.json")
                or str(child.get("name", "")).endswith("_channels.tsv")
            )
        ]
        if sum(str(item.get("name", "")).endswith("_eeg.set") for item in wanted) != 1:
            raise ValueError(f"Expected one raw EEG .set file for {participant_id}.")
        for item in wanted:
            name = str(item["name"])
            size = int(item["size"]) if item.get("size") is not None else None
            relative = Path(participant_id) / "eeg" / name
            downloads.append((f"{BASE_URL}/{participant_id}/eeg/{quote(name)}", root / relative, size))

    total_bytes = sum(size or 0 for _, _, size in downloads)
    completed = 0
    errors: list[Exception] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 6))) as executor:
        futures = {
            executor.submit(_download, url, path, size): path
            for url, path, size in downloads
        }
        for future in as_completed(futures):
            try:
                future.result()
                completed += 1
                if completed % 12 == 0 or completed == len(futures):
                    print(
                        f"Downloaded/verified {completed}/{len(futures)} BIDS files "
                        f"({total_bytes / (1024**3):.2f} GiB raw payload).",
                        flush=True,
                    )
            except Exception as exc:
                errors.append(exc)
    if errors:
        first = errors[0]
        raise RuntimeError(
            f"{len(errors)} dataset files failed to download; first failure: {first}"
        ) from first

    frame, inventory = load_dataset_index(root)
    manifest = {
        **inventory,
        "downloaded_files": len(downloads) + len(METADATA_FILES),
        "downloaded_bytes": int(
            sum(path.stat().st_size for _, path, _ in downloads)
            + sum((root / name).stat().st_size for name in METADATA_FILES)
        ),
        "source_url": f"https://www.nemar.org/dataset/{NEMAR_DATASET_ID}",
        "retrieved_from": BASE_URL,
        "derivatives_downloaded": False,
    }
    manifest_path = root / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Verified {len(frame)} participants: {inventory['class_counts']}; "
        f"{inventory['raw_eeg_bytes'] / (1024**3):.2f} GiB raw EEG.",
        flush=True,
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/openneuro_ds004504")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    download_dataset(args.output, args.workers)


if __name__ == "__main__":
    main()
