import torch

from anomdet.model import MLSTMAutoencoder


def test_output_shape_1d():
    model = MLSTMAutoencoder(
        seq_len=32, in_dim=1, d_model=16, latent_dim=4, enc_blocks=1, dec_blocks=1
    )
    x = torch.randn(4, 32, 1)
    assert model(x).shape == x.shape


def test_output_shape_2d():
    model = MLSTMAutoencoder(
        seq_len=24, in_dim=2, d_model=16, latent_dim=4, enc_blocks=1, dec_blocks=1
    )
    x = torch.randn(2, 24, 2)
    assert model(x).shape == x.shape


def test_latent_bottleneck():
    # latent_dim < d_model forces actual compression
    model = MLSTMAutoencoder(
        seq_len=16, in_dim=1, d_model=32, latent_dim=2, enc_blocks=1, dec_blocks=1
    )
    x = torch.randn(3, 16, 1)
    out = model(x)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)  # model actually transforms the input
