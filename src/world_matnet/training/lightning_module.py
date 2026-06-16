"""LightningModule for quantile training."""

from __future__ import annotations

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl

import torch

from world_matnet.models.model_quantile import AgriMatNetQuantile, quantile_loss
from world_matnet.training.batch import apply_ablation, masked_time_weighted_mse
from world_matnet.training.utils import masked_mse


def _masked_mae_rmse(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    keep = ~mask
    if not keep.any():
        zero = torch.tensor(0.0, device=preds.device)
        return zero, zero
    diff = preds[keep] - targets[keep]
    mae = diff.abs().mean()
    rmse = torch.sqrt((diff * diff).mean() + 1e-12)
    return mae, rmse


class QuantileLightningModule(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        quantiles: list[float],
        lr: float = 1e-3,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        ablation_cfg: dict | None = None,
        feature_idx: dict | None = None,
        time_weight_alpha: float | None = None,
        lr_reduce_on_plateau: bool = False,
        lr_factor: float = 0.2,
        lr_patience: int = 10,
        lr_min: float = 5e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = AgriMatNetQuantile(
            input_dim=input_dim,
            quantiles=quantiles,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.quantiles = list(quantiles)
        self.median_idx = len(self.quantiles) // 2
        self.ablation_cfg = ablation_cfg or {
            "future_covariates_off": False,
            "history_covariates_off": False,
            "target_history_off": False,
        }
        self.feature_idx = feature_idx

    def forward(self, batch):
        return self.model(batch)

    def _compute_losses(self, batch):
        batch = apply_ablation(batch, self.ablation_cfg, self.feature_idx)
        preds = self.model(batch)
        targets = batch["target"]
        mask = batch["target_mask"]

        weights = None
        delta_days = None
        if self.hparams.time_weight_alpha is not None:
            delta_days = batch["target_delta_days"]
            weights = 1.0 / (1.0 + self.hparams.time_weight_alpha * delta_days)

        loss = quantile_loss(preds, targets, mask, self.quantiles, weights=weights)
        if self.hparams.time_weight_alpha is not None:
            mse = masked_time_weighted_mse(
                preds[..., self.median_idx],
                targets,
                mask,
                delta_days,
                alpha=self.hparams.time_weight_alpha,
            )
        else:
            mse = masked_mse(preds[..., self.median_idx], targets, mask)

        mae, rmse = _masked_mae_rmse(preds[..., self.median_idx], targets, mask)

        metrics = {
            "loss": loss,
            "quantile_loss": loss,
            "pinball_loss": loss,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
        }
        return loss, metrics

    def _log_metrics(self, stage: str, metrics: dict[str, torch.Tensor], prog_bar: bool = False):
        # New unified taxonomy.
        new_names = {f"{stage}/{k}": v for k, v in metrics.items()}
        self.log_dict(new_names, on_step=False, on_epoch=True, prog_bar=prog_bar, sync_dist=False)

        # Legacy aliases kept for backward compatibility with old dashboards.
        if stage in {"train", "val", "test"}:
            self.log(f"{stage}_pinball", metrics["pinball_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
            self.log(f"{stage}_mse", metrics["mse"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

    def training_step(self, batch, batch_idx):
        loss, metrics = self._compute_losses(batch)
        self._log_metrics("train", metrics, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, metrics = self._compute_losses(batch)
        self._log_metrics("val", metrics, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, metrics = self._compute_losses(batch)
        self._log_metrics("test", metrics, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        if not self.hparams.lr_reduce_on_plateau:
            return optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.lr_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
