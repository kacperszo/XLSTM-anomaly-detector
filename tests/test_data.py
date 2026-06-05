from unittest.mock import patch

import numpy as np
import torch

from anomdet.data import load_dataset, make_loader


def _fake_load(name, split, **kwargs):
    X = np.random.randn(10, 1, 20).astype(np.float32)
    y = np.array(["1"] * 7 + ["-1"] * 3)
    return X, y


def test_load_no_dwt():
    with patch("anomdet.data.load_classification", side_effect=_fake_load):
        X, y = load_dataset("Fake", normal_label="1", use_dwt=False)
    assert X.shape == (10, 20, 1)
    assert int(y.sum()) == 3


def test_load_with_dwt():
    with patch("anomdet.data.load_classification", side_effect=_fake_load):
        X, y = load_dataset("Fake", normal_label="1", use_dwt=True, wavelet="db4")
    assert X.shape[0] == 10
    assert X.shape[2] == 2  # two DWT channels


def test_make_loader_normal_only():
    X = torch.randn(10, 20, 1)
    y = np.array([0] * 7 + [1] * 3)
    loader = make_loader(X, y, batch_size=4, device=torch.device("cpu"), normal_only=True)
    total = sum(b[0].size(0) for b in loader)
    assert total == 7
