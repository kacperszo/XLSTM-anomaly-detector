from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def plot_reconstruction(
    model: torch.nn.Module,
    X_tensor: torch.Tensor,
    y_binary: np.ndarray,
    index: int,
    device: torch.device,
    top_k_percent: float = 15.0,
    save_path: Path | None = None,
) -> None:
    """Two-panel reconstruction error plot with hotspot shading."""
    model.eval()
    x = X_tensor[index].unsqueeze(0).to(device)
    x_recon = model(x)

    orig = x.squeeze().cpu().numpy()
    recon = x_recon.squeeze().cpu().numpy()

    if orig.ndim == 2:  # multi-channel: take first channel for display
        orig, recon = orig[:, 0], recon[:, 0]

    mse_per_step = (orig - recon) ** 2
    threshold = np.percentile(mse_per_step, 100 - top_k_percent)
    hotspot = mse_per_step > threshold

    is_anomaly = bool(y_binary[index])
    color = "#d62728" if is_anomaly else "#2ca02c"
    label = "Anomaly" if is_anomaly else "Normal"

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    ax1.plot(orig, color=color, linewidth=2, label=f"Input ({label})")
    ax1.plot(
        recon, color="#1f77b4", linestyle="--", linewidth=1.5, label="Reconstruction"
    )
    ax1.fill_between(
        range(len(orig)),
        orig.min(),
        orig.max(),
        where=hotspot,
        color="#ff7f0e",
        alpha=0.25,
        label=f"Top {top_k_percent:.0f}% error",
    )
    ax1.set_title(f"xLSTM Reconstruction - sample #{index}", fontsize=14)
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.bar(
        range(len(mse_per_step)), mse_per_step, color="#ff7f0e", alpha=0.8, width=1.0
    )
    ax2.axhline(threshold, color="red", linestyle=":", linewidth=1.5, label="threshold")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("MSE")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def shap_timeseries(
    model: torch.nn.Module,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    import shap

    orig_shape = X_background.shape[1:]  # (T,) or (T, C)
    bg_flat = X_background.reshape(X_background.shape[0], -1)
    ex_flat = X_explain.reshape(X_explain.shape[0], -1)

    def _score(x_flat: np.ndarray) -> np.ndarray:
        x_np = x_flat.reshape(-1, *orig_shape)
        if x_np.ndim == 2:  # single-channel: add C dim
            x_np = x_np[:, :, np.newaxis]
        model.eval()
        all_scores: list[float] = []
        bs = 128
        for i in range(0, len(x_np), bs):
            xb = torch.tensor(x_np[i : i + bs], dtype=torch.float32).to(device)
            with torch.no_grad():
                xr = model(xb)
                s = (xb - xr).pow(2).mean(dim=tuple(range(1, xb.ndim)))
            all_scores.extend(s.cpu().numpy().tolist())
        return np.array(all_scores)

    explainer = shap.KernelExplainer(_score, bg_flat)
    sv_flat = cast(np.ndarray, explainer.shap_values(ex_flat))
    return sv_flat.reshape(X_explain.shape[0], *orig_shape)
