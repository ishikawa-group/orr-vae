#!/usr/bin/env python3
"""Generate condition_list.csv for Pt-Ni batch submissions."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path


label_thresholds = [0.15, 0.3]
batch_sizes = [16]
max_epochs = [100, 200]
latent_sizes = [16, 32]


def main() -> int:
    combinations = list(itertools.product(label_thresholds, batch_sizes, max_epochs, latent_sizes))
    output_file = Path(__file__).resolve().parent / "condition_list.csv"

    with output_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "label_threshold", "batch_size", "max_epoch", "latent_size"])
        for i, (label_threshold, batch_size, max_epoch, latent_size) in enumerate(combinations, 1):
            writer.writerow([i, label_threshold, batch_size, max_epoch, latent_size])

    print(f"saved: {output_file}")
    print(f"total conditions: {len(combinations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
