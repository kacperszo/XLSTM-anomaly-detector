from __future__ import annotations

import warnings

import numpy as np
import pywt
import torch
from aeon.datasets import load_classification
from torch.utils.data import DataLoader, TensorDataset


def _load_raw(name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, y = load_classification(
            name,
            split=split,
            load_equal_length=True,
            load_no_missing=True,
        )
    return X, y


def load_dataset(
    name: str,
    split: str = "train",
    normal_label: str = "1",
    use_dwt: bool = False,
    wavelet: str = "db4",
) -> tuple[torch.Tensor, np.ndarray]:
    """Load an aeon classification dataset as anomaly-detection tensors.

    Returns
    -------
    X : torch.Tensor  shape (n, T, C)
    y_binary : np.ndarray  0 = normal, 1 = anomaly
    """
    X_raw, y_raw = _load_raw(name, split)

    if use_dwt:
        X_flat = X_raw.squeeze(1)  # (n, T)
        cA, cD = pywt.dwt(X_flat, wavelet, axis=1)
        X_proc = np.stack([cA, cD], axis=-1).astype(np.float32)  # (n, T//2+1, 2)
    else:
        X_proc = X_raw.transpose(0, 2, 1).astype(np.float32)  # (n, T, 1)

    y_binary = np.where(y_raw == normal_label, 0, 1).astype(np.int64)
    return torch.from_numpy(X_proc), y_binary


def make_loader(
    X: torch.Tensor,
    y_binary: np.ndarray,
    batch_size: int,
    device: torch.device,
    normal_only: bool = True,
    shuffle: bool = True,
) -> DataLoader[tuple[torch.Tensor, ...]]:
    if normal_only:
        X = X[y_binary == 0]
    dataset = TensorDataset(X.to(device))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
