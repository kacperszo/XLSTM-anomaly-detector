import numpy as np

from anomdet.evaluate import compute_scores, evaluate


def test_perfect_separation():
    scores = np.array([0.1, 0.15, 0.85, 0.9])
    y = np.array([0, 0, 1, 1])
    m = evaluate(scores, y)
    assert m["auroc"] == 1.0
    assert m["auprc"] == 1.0
    assert 0.15 <= m["threshold"] <= 0.85


def test_random_scores_auroc_range():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 100)
    y = rng.integers(0, 2, 100)
    m = evaluate(scores, y)
    assert 0.0 <= m["auroc"] <= 1.0
    assert 0.0 <= m["auprc"] <= 1.0


def test_compute_scores_mse_shape():
    import torch

    from anomdet.model import MLSTMAutoencoder

    model = MLSTMAutoencoder(
        seq_len=16, in_dim=1, d_model=16, latent_dim=4, enc_blocks=1, dec_blocks=1
    )
    X = torch.randn(10, 16, 1)
    device = torch.device("cpu")
    scores = compute_scores(model, X, device)
    assert scores.shape == (10,)
    assert (scores >= 0).all()
