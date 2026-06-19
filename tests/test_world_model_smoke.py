"""Synthetic smoke tests for the modular vegetation world model."""

from __future__ import annotations

import unittest

import numpy as np
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenario import perturb_future_covariates
from training.batch import collate_variable
from training.world_lightning_module import VegetationWorldLightningModule
from models.world_model import VegetationWorldModel


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


def _build_batch() -> dict:
    item1 = _make_item(hist_len=6, fut_len=8, positions=[1, 4, 7], values=[0.3, 0.4, 0.5])
    item2 = _make_item(hist_len=5, fut_len=7, positions=[0, 3, 6], values=[0.2, 0.1, 0.7])
    return collate_variable([item1, item2])


class WorldModelSmokeTests(unittest.TestCase):
    def test_forward_shapes_for_all_dynamics(self):
        batch = _build_batch()
        bsz = batch["history"].shape[0]
        full_len = batch["future"].shape[1]

        for dynamics_type in ["residual_mlp", "gru", "transformer", "rssm"]:
            model = VegetationWorldModel(
                input_dim=batch["history"].shape[-1],
                target_dim=1,
                quantiles=[0.1, 0.5, 0.9],
                d_model=32,
                num_layers=1,
                num_heads=4,
                dim_feedforward=64,
                dropout=0.1,
                dynamics_type=dynamics_type,
                latent_dim=32,
                dynamics_hidden_dim=32,
                rssm_deterministic_dim=32,
                rssm_stochastic_dim=32,
                rssm_hidden_dim=32,
            )
            out = model(batch)
            self.assertEqual(out["quantiles"].shape, (bsz, full_len, 1, 3))
            self.assertEqual(out["latent_states"].shape, (bsz, full_len, 32))

    def test_sparse_supervision_scatter(self):
        batch = _build_batch()
        dense = batch["target_dense"]
        dense_mask = batch["target_dense_mask"]

        # Sample 0 has supervision at positions [1,4,7]
        self.assertFalse(bool(dense_mask[0, 1, 0]))
        self.assertFalse(bool(dense_mask[0, 4, 0]))
        self.assertFalse(bool(dense_mask[0, 7, 0]))
        self.assertTrue(bool(dense_mask[0, 0, 0]))

        # Sample 1 has padded last step in this batch (max future is 8, length is 7)
        self.assertTrue(bool(batch["future_pad_mask"][1, 7]))
        self.assertTrue(bool(dense_mask[1, 7, 0]))

        # Value consistency on scattered points
        self.assertAlmostEqual(float(dense[0, 1, 0]), 0.3, places=6)
        self.assertAlmostEqual(float(dense[0, 4, 0]), 0.4, places=6)
        self.assertAlmostEqual(float(dense[0, 7, 0]), 0.5, places=6)

    def test_loss_is_finite(self):
        batch = _build_batch()
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
            smoothness_loss_weight=0.01,
            aux_loss_weight=0.01,
            use_aux_decoder=True,
            sigreg_weight=1e-3,
            use_sigreg_projector=True,
            sigreg_projector_num_layers=1,
            sigreg_projector_output_dim=32,
            sigreg_num_proj=64,
        )
        loss, metrics, _ = module._compute_loss_and_metrics(batch)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIn("quantile_loss", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("sigreg_loss", metrics)

    def test_scenario_perturbation_shape_and_columns(self):
        batch = _build_batch()
        x_future = batch["future"]
        feature_names = [
            "wind",
            "rainfall",
            "avg_temperature",
            "humidity",
            "solar",
            "target_history",
        ]
        cfg = {"precipitation_multiplier": 0.8, "temperature_additive": 2.0}
        perturbed, _ = perturb_future_covariates(
            x_future=x_future,
            feature_names=feature_names,
            scenario_config=cfg,
            future_mask=batch["future_mask"],
        )

        self.assertEqual(tuple(perturbed.shape), tuple(x_future.shape))

        # Rainfall and temperature columns changed; wind unchanged.
        self.assertTrue(torch.any(torch.ne(perturbed[..., 1], x_future[..., 1])).item())
        self.assertTrue(torch.any(torch.ne(perturbed[..., 2], x_future[..., 2])).item())
        self.assertTrue(torch.allclose(perturbed[..., 0], x_future[..., 0]))

    def test_inference_without_y_future(self):
        batch = _build_batch()
        model = VegetationWorldModel(
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
        model.eval()

        batch_infer = dict(batch)
        batch_infer.pop("target_dense")
        batch_infer.pop("target_dense_mask")

        with torch.no_grad():
            out = model(batch_infer)

        self.assertEqual(out["quantiles"].shape[:3], (batch["history"].shape[0], batch["future"].shape[1], 1))


if __name__ == "__main__":
    unittest.main()
