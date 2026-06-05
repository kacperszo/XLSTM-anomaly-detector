from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from xlstm import (
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
)


def _mlstm_stack(d_model: int, n_blocks: int, ctx_len: int) -> xLSTMBlockStack:
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4,
                qkv_proj_blocksize=4,
                num_heads=4,
            )
        ),
        context_length=ctx_len,
        num_blocks=n_blocks,
        embedding_dim=d_model,
        slstm_at=[],
    )
    return xLSTMBlockStack(cfg)


class MLSTMAutoencoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        in_dim: int = 1,
        d_model: int = 64,
        latent_dim: int = 16,
        enc_blocks: int = 2,
        dec_blocks: int = 2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_proj = nn.Linear(in_dim, d_model)
        self.encoder = _mlstm_stack(d_model, enc_blocks, seq_len)
        self.to_z = nn.Linear(d_model, latent_dim)
        self.from_z = nn.Linear(latent_dim, d_model)
        self.decoder = _mlstm_stack(d_model, dec_blocks, seq_len)
        self.out_proj = nn.Linear(d_model, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = cast(torch.Tensor, self.encoder(h))
        z = self.to_z(h.mean(dim=1))
        d = self.from_z(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        d = cast(torch.Tensor, self.decoder(d))
        return cast(torch.Tensor, self.out_proj(d))
