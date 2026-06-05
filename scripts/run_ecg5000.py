#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import torch
import yaml

from anomdet.data import load_dataset, make_loader
from anomdet.evaluate import compute_scores, evaluate
from anomdet.model import MLSTMAutoencoder
from anomdet.train import train


def run(config_path: str) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    load_kw = dict(
        normal_label=cfg["normal_label"],
        use_dwt=cfg.get("use_dwt", False),
        wavelet=cfg.get("wavelet", "db4"),
    )
    X_train, y_train = load_dataset(cfg["dataset"], split="train", **load_kw)
    X_test, y_test = load_dataset(cfg["dataset"], split="test", **load_kw)

    seq_len, in_dim = X_train.shape[1], X_train.shape[2]

    model = MLSTMAutoencoder(seq_len=seq_len, in_dim=in_dim, **cfg["model"]).to(device)
    loader = make_loader(X_train, y_train, cfg["train"]["batch_size"], device)

    train(
        model,
        loader,
        epochs=cfg["train"]["epochs"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        device=device,
    )

    scores = compute_scores(model, X_test, device)
    metrics = evaluate(scores, y_test)
    print(
        f"\nAUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  threshold={metrics['threshold']:.4f}"
    )

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    tag = cfg["dataset"].lower()
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save(model.state_dict(), out_dir / f"{tag}_model.pt")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ecg5000.yaml")
    args = parser.parse_args()
    run(args.config)
