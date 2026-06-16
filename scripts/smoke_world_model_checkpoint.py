#!/usr/bin/env python3
"""Checkpoint smoke check for world_model_gru LightningModule."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from world_matnet.training.batch import collate_variable
from world_matnet.training.world_lightning_module import VegetationWorldLightningModule


def _make_item(hist_len: int, fut_len: int, positions: list[int], values: list[float], feature_dim: int = 6):
    rng = np.random.default_rng(seed=hist_len * 100 + fut_len)

    history = rng.normal(size=(hist_len, feature_dim)).astype(np.float32)
    future = rng.normal(size=(fut_len, feature_dim)).astype(np.float32)

    history_mask = np.zeros_like(history, dtype=bool)
    future_mask = np.zeros_like(future, dtype=bool)
    future_noise = np.zeros_like(future, dtype=np.float32)

    target = np.array(values, dtype=np.float32)
    target_mask = np.zeros_like(target, dtype=bool)

    base = np.datetime64("2020-01-01")
    history_ts = [str(base + np.timedelta64(i, "D")) for i in range(hist_len)]
    future_ts = [str(base + np.timedelta64(hist_len + i + 1, "D")) for i in range(fut_len)]
    target_ts = [future_ts[p] for p in positions]

    return {
        "history": torch.tensor(history, dtype=torch.float32),
        "history_mask": torch.tensor(history_mask, dtype=torch.bool),
        "history_timestamps": history_ts,
        "future": torch.tensor(future, dtype=torch.float32),
        "future_mask": torch.tensor(future_mask, dtype=torch.bool),
        "future_noise": torch.tensor(future_noise, dtype=torch.float32),
        "future_timestamps": future_ts,
        "future_target_positions": torch.tensor(positions, dtype=torch.long),
        "target": torch.tensor(target, dtype=torch.float32),
        "target_mask": torch.tensor(target_mask, dtype=torch.bool),
        "target_timestamps": target_ts,
        "climate": "temperate",
        "latitude": 45.0,
        "longitude": 9.0,
        "crop_type": "wheat",
        "area": "A001",
        "source": "synthetic.csv",
        "history_start": history_ts[0],
        "history_end": history_ts[-1],
        "future_start": future_ts[0],
        "future_end": future_ts[-1],
    }


class _SyntheticDataset(Dataset):
    def __init__(self):
        self.items = [
            _make_item(hist_len=6, fut_len=8, positions=[1, 4, 7], values=[0.3, 0.4, 0.5]),
            _make_item(hist_len=5, fut_len=7, positions=[0, 3, 6], values=[0.2, 0.1, 0.7]),
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def main():
    pl.seed_everything(42, workers=True)

    dataset = _SyntheticDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_variable)
    batch = next(iter(loader))

    module = VegetationWorldLightningModule(
        input_dim=batch["history"].shape[-1],
        target_dim=1,
        quantiles=[0.1, 0.5, 0.9],
        d_model=32,
        num_layers=1,
        num_heads=4,
        dim_feedforward=64,
        dropout=0.1,
        dynamics_type="gru",
        latent_dim=32,
        dynamics_hidden_dim=32,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "world_model_gru.ckpt"
        trainer.save_checkpoint(str(ckpt_path))
        restored = VegetationWorldLightningModule.load_from_checkpoint(str(ckpt_path), map_location="cpu")
        restored.eval()
        with torch.no_grad():
            out = restored(batch)
        assert out["quantiles"].shape[-1] == 3

    print("Checkpoint roundtrip smoke check passed")


if __name__ == "__main__":
    main()
