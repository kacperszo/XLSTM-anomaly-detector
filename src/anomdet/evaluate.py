from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset


@torch.no_grad()
def compute_scores(
    model: torch.nn.Module,
    X: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    chunks: list[np.ndarray] = []

    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        x_recon = model(x_batch)
        s = (x_batch - x_recon).pow(2).mean(dim=tuple(range(1, x_batch.ndim)))
        chunks.append(s.cpu().numpy())

    return np.concatenate(chunks)


def evaluate(scores: np.ndarray, y_binary: np.ndarray) -> dict[str, float]:
    auroc = float(roc_auc_score(y_binary, scores))
    auprc = float(average_precision_score(y_binary, scores))

    fpr, tpr, thresholds = roc_curve(y_binary, scores)
    best_thresh = float(thresholds[np.argmax(tpr - fpr)])

    return {"auroc": auroc, "auprc": auprc, "threshold": best_thresh}
