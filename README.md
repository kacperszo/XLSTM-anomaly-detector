# ECG Anomaly Detection with xLSTM

Unsupervised anomaly detection in time series using an **mLSTM autoencoder**. The model is trained exclusively on normal samples; anomaly score at inference time is the reconstruction error.

## Problem

Given a labelled time-series benchmark (ECG, engine vibration, wafer sensor), identify anomalous signals without using anomaly labels during training. Evaluation uses held-out test splits with AUROC / AUPRC.

## Method

```
Input (n, T, C)
    ↓  Linear projection
    ↓  mLSTM encoder  (2 blocks)
    ↓  mean-pool → latent z  (dim 8–16)
    ↓  repeat → mLSTM decoder (2 blocks)
    ↓  Linear projection
Reconstruction (n, T, C)

Anomaly score = MSE(x, x̂)
```

Threshold is selected by maximising Youden's J on the validation set.

## Setup

```bash
uv sync
make download       # pre-fetch all datasets
```

## Reproduce

```bash
make run-ecg5000    # -> results/ecg5000_metrics.json
make run-forda      # -> results/forda_metrics.json
make run-wafer      # -> results/wafer_metrics.json
make test           # sanity tests
```

Override any hyperparameter via a custom config:

```bash
uv run python scripts/run_ecg5000.py --config configs/ecg5000.yaml
```

## Project layout

```
src/anomdet/
  data.py      loaders, DWT preprocessing
  model.py     MLSTMAutoencoder
  train.py     training loop (AdamW + grad-clip)
  evaluate.py  AUROC, AUPRC, threshold selection
  explain.py   reconstruction error map, KernelSHAP
scripts/
  run_ecg5000.py / run_forda.py
configs/       YAML per experiment
notebooks/     exploratory notebooks
tests/         sanity checks
```
