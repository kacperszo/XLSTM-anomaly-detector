from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
) -> list[float]:
    if device is None:
        device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    model.train()
    loss_history: list[float] = []

    for _ in tqdm(range(epochs), desc="Training", unit="ep"):
        epoch_loss = 0.0

        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), X_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        loss_history.append(epoch_loss / len(loader.dataset))  # type: ignore[arg-type]

    return loss_history
