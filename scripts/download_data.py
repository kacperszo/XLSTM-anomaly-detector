#!/usr/bin/env python
import warnings

from aeon.datasets import load_classification

DATASETS = ["ECG5000", "FordA", "Wafer"]

for name in DATASETS:
    for split in ("train", "test"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X, y = load_classification(
                name, split=split, load_equal_length=True, load_no_missing=True
            )
        print(f"  {name:10s} {split:5s}  {X.shape}")

print("All datasets ready.")
