"""LightningModule for modular vegetation world-model training."""

from __future__ import annotations

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_matnet.models.world_model import (
    SketchedIsotropicGaussianRegularizer,
    VegetationWorldModel,
    calibration_metrics,
    latent_smoothness_loss,
    masked_kl_loss,
    masked_mae_rmse,
    masked_regression_loss,
    noncrossing_loss,
    weighted_quantile_loss_dense,
)


def _build_sigreg_projector(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Sequential:
    """Builds an MLP projector with BatchNorm for SIGReg branch."""
    if num_layers < 1:
        raise ValueError("sigreg_projector_num_layers must be >= 1")

    layers: list[nn.Module] = []
    if num_layers == 1:
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))
        return nn.Sequential(*layers)

    layers.extend([nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
    for _ in range(num_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
    layers.extend([nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim)])
    return nn.Sequential(*layers)


class VegetationWorldLightningModule(pl.LightningModule):
    """Lightning wrapper around ``VegetationWorldModel`` with modular losses."""

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        quantiles: list[float],
        lr: float = 1e-3,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        summarizer_type: str = "attention_pooling",
        latent_dim: int = 128,
        sample_initial_state: bool = True,
        initial_state_kl_weight: float = 0.0,
        use_horizon_embedding: bool = True,
        horizon_emb_dim: int = 32,
        horizon_embedding_type: str = "learned",
        max_horizon: int = 512,
        spatial_enabled: bool = False,
        spatial_cont_dim: int = 0,
        spatial_cat_cardinalities: list[int] | None = None,
        spatial_emb_dim: int = 32,
        use_harmonic_coords: bool = False,
        num_harmonic_frequencies: int = 4,
        spatial_dropout: float = 0.1,
        dynamics_type: str = "gru",
        dynamics_hidden_dim: int = 128,
        dynamics_num_layers: int = 2,
        use_residual_update: bool = False,
        rssm_deterministic_dim: int = 128,
        rssm_stochastic_dim: int = 128,
        rssm_hidden_dim: int = 128,
        sample_latent: bool = True,
        decoder_hidden_dim: int = 128,
        decoder_num_layers: int = 2,
        use_horizon_conditioned_decoder: bool = False,
        enforce_quantile_ordering: bool = False,
        use_risk_head: bool = False,
        risk_threshold: float = 0.2,
        risk_loss_weight: float = 0.0,
        smoothness_loss_weight: float = 0.0,
        smoothness_loss_type: str = "l2",
        aux_loss_weight: float = 0.0,
        aux_loss_type: str = "mae",
        use_aux_decoder: bool = False,
        rssm_kl_weight: float = 0.0,
        rssm_free_nats: float = 0.0,
        rssm_kl_warmup_epochs: int = 0,
        posterior_mixing_enabled: bool = False,
        posterior_mixing_start_prob: float = 1.0,
        posterior_mixing_end_prob: float = 1.0,
        posterior_mixing_anneal_epochs: int = 0,
        noncrossing_loss_weight: float = 0.0,
        compute_calibration_metrics: bool = False,
        sigreg_weight: float = 0.0,
        sigreg_num_proj: int = 256,
        sigreg_knots: int = 17,
        use_sigreg_projector: bool = False,
        sigreg_projector_hidden_dim: int = 128,
        sigreg_projector_output_dim: int | None = None,
        sigreg_projector_num_layers: int = 1,
        sigreg_projector_dropout: float = 0.0,
        time_weight_alpha: float | None = None,
        lr_reduce_on_plateau: bool = False,
        lr_factor: float = 0.2,
        lr_patience: int = 10,
        lr_min: float = 5e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = VegetationWorldModel(
            input_dim=input_dim,
            target_dim=target_dim,
            quantiles=quantiles,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            summarizer_type=summarizer_type,
            latent_dim=latent_dim,
            sample_initial_state=sample_initial_state,
            use_horizon_embedding=use_horizon_embedding,
            horizon_emb_dim=horizon_emb_dim,
            horizon_embedding_type=horizon_embedding_type,
            max_horizon=max_horizon,
            spatial_enabled=spatial_enabled,
            spatial_cont_dim=spatial_cont_dim,
            spatial_cat_cardinalities=spatial_cat_cardinalities,
            spatial_emb_dim=spatial_emb_dim,
            use_harmonic_coords=use_harmonic_coords,
            num_harmonic_frequencies=num_harmonic_frequencies,
            spatial_dropout=spatial_dropout,
            dynamics_type=dynamics_type,
            dynamics_hidden_dim=dynamics_hidden_dim,
            dynamics_num_layers=dynamics_num_layers,
            use_residual_update=use_residual_update,
            rssm_deterministic_dim=rssm_deterministic_dim,
            rssm_stochastic_dim=rssm_stochastic_dim,
            rssm_hidden_dim=rssm_hidden_dim,
            sample_latent=sample_latent,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_num_layers=decoder_num_layers,
            use_horizon_conditioned_decoder=use_horizon_conditioned_decoder,
            enforce_quantile_ordering=enforce_quantile_ordering,
            use_risk_head=use_risk_head,
        )

        self.quantiles = list(quantiles)
        self.median_idx = len(self.quantiles) // 2

        self.aux_decoder: nn.Module | None = None
        if use_aux_decoder:
            self.aux_decoder = nn.Sequential(
                nn.Linear(latent_dim, decoder_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(decoder_hidden_dim, target_dim),
            )

        self.sigreg: SketchedIsotropicGaussianRegularizer | None = None
        if float(sigreg_weight) > 0.0:
            self.sigreg = SketchedIsotropicGaussianRegularizer(
                knots=int(sigreg_knots),
                num_proj=int(sigreg_num_proj),
            )

        self.sigreg_projector: nn.Module | None = None
        if use_sigreg_projector:
            sigreg_out_dim = int(sigreg_projector_output_dim or latent_dim)
            self.sigreg_projector = _build_sigreg_projector(
                input_dim=int(latent_dim),
                hidden_dim=int(sigreg_projector_hidden_dim),
                output_dim=sigreg_out_dim,
                num_layers=int(sigreg_projector_num_layers),
                dropout=float(sigreg_projector_dropout),
            )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model(batch)

    def _rssm_kl_weight_effective(self) -> float:
        """Returns KL weight after optional linear warmup."""
        target = float(self.hparams.rssm_kl_weight)
        if target <= 0.0 or self.hparams.dynamics_type != "rssm":
            return target

        warmup_epochs = int(max(0, self.hparams.rssm_kl_warmup_epochs))
        if warmup_epochs <= 0:
            return target

        progress = min(1.0, max(0.0, float(self.current_epoch) / float(warmup_epochs)))
        return target * progress

    def _posterior_use_probability(self) -> float | None:
        """Returns probability of using posterior latent for supervised RSSM steps."""
        if self.hparams.dynamics_type != "rssm" or not bool(self.hparams.posterior_mixing_enabled):
            return None

        start = float(self.hparams.posterior_mixing_start_prob)
        end = float(self.hparams.posterior_mixing_end_prob)
        start = min(1.0, max(0.0, start))
        end = min(1.0, max(0.0, end))

        anneal_epochs = int(max(0, self.hparams.posterior_mixing_anneal_epochs))
        if anneal_epochs <= 0:
            return end

        progress = min(1.0, max(0.0, float(self.current_epoch) / float(anneal_epochs)))
        return start + (end - start) * progress

    def _temporal_weights(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        alpha = self.hparams.time_weight_alpha
        if alpha is None:
            return None
        if "future_delta_days" not in batch:
            return None

        delta_days = torch.clamp(batch["future_delta_days"], min=0.0)
        return 1.0 / (1.0 + float(alpha) * delta_days)

    @staticmethod
    def _valid_latent_tokens(latent_states: torch.Tensor, future_pad_mask: torch.Tensor) -> torch.Tensor | None:
        """Returns valid latent tokens as [N_valid, D] (or None if insufficient)."""
        if latent_states.dim() != 3 or future_pad_mask.dim() != 2:
            raise ValueError("Expected latent_states [B,L,D] and future_pad_mask [B,L]")

        keep = ~future_pad_mask
        if not keep.any():
            return None

        flat_latent = latent_states.reshape(-1, latent_states.shape[-1])
        flat_keep = keep.reshape(-1)
        tokens = flat_latent[flat_keep]
        if tokens.shape[0] < 2:
            return None
        return tokens

    def _compute_sigreg_loss(self, latent_states: torch.Tensor, future_pad_mask: torch.Tensor) -> torch.Tensor:
        if self.sigreg is None:
            return torch.zeros((), device=latent_states.device, dtype=latent_states.dtype)

        tokens = self._valid_latent_tokens(latent_states, future_pad_mask)
        if tokens is None:
            return torch.zeros((), device=latent_states.device, dtype=latent_states.dtype)

        if self.sigreg_projector is not None:
            # BatchNorm projector requires >=2 samples in training mode.
            if self.training and tokens.shape[0] < 2:
                return torch.zeros((), device=latent_states.device, dtype=latent_states.dtype)
            tokens = self.sigreg_projector(tokens)

        return self.sigreg(tokens)

    def _compute_loss_and_metrics(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        posterior_use_prob = self._posterior_use_probability()
        out = self.model(batch, posterior_use_probability=posterior_use_prob)

        preds_q = out["quantiles"]
        targets = batch["target_dense"]
        target_mask = batch["target_dense_mask"]
        future_pad_mask = batch["future_pad_mask"]

        weights = self._temporal_weights(batch)
        quantile_loss = weighted_quantile_loss_dense(
            preds=preds_q,
            targets=targets,
            mask=target_mask,
            quantiles=self.quantiles,
            weights=weights,
        )

        total_loss = quantile_loss
        metrics: dict[str, torch.Tensor] = {
            "quantile_loss": quantile_loss,
            "pinball_loss": quantile_loss,
        }

        median_pred = preds_q[..., self.median_idx]
        mae, rmse = masked_mae_rmse(median_pred, targets, target_mask)
        metrics["mae"] = mae
        metrics["rmse"] = rmse

        if self.hparams.smoothness_loss_weight > 0.0:
            smooth = latent_smoothness_loss(
                out["latent_states"],
                future_pad_mask=future_pad_mask,
                loss_type=self.hparams.smoothness_loss_type,
            )
            total_loss = total_loss + float(self.hparams.smoothness_loss_weight) * smooth
            metrics["smoothness_loss"] = smooth

        if self.hparams.aux_loss_weight > 0.0:
            if self.aux_decoder is not None:
                aux_pred = self.aux_decoder(out["latent_states"])
            else:
                aux_pred = median_pred
            aux = masked_regression_loss(
                preds=aux_pred,
                targets=targets,
                mask=target_mask,
                loss_type=self.hparams.aux_loss_type,
            )
            total_loss = total_loss + float(self.hparams.aux_loss_weight) * aux
            metrics["aux_loss"] = aux

        if self.hparams.sigreg_weight > 0.0:
            sigreg_loss = self._compute_sigreg_loss(
                latent_states=out["latent_states"],
                future_pad_mask=future_pad_mask,
            )
            total_loss = total_loss + float(self.hparams.sigreg_weight) * sigreg_loss
            metrics["sigreg_loss"] = sigreg_loss

        if self.hparams.initial_state_kl_weight > 0.0 and "z0_mu" in out and "z0_logvar" in out:
            z0_mu = out["z0_mu"]
            z0_logvar = out["z0_logvar"]
            z0_kl = 0.5 * (z0_mu.pow(2) + z0_logvar.exp() - z0_logvar - 1.0).mean()
            total_loss = total_loss + float(self.hparams.initial_state_kl_weight) * z0_kl
            metrics["initial_state_kl_loss"] = z0_kl

        if self.hparams.rssm_kl_weight > 0.0 and self.hparams.dynamics_type == "rssm":
            if all(k in out for k in ["prior_mu", "prior_logvar", "posterior_mu", "posterior_logvar"]):
                kl_weight = self._rssm_kl_weight_effective()
                kl = masked_kl_loss(
                    prior_mu=out["prior_mu"],
                    prior_logvar=out["prior_logvar"],
                    post_mu=out["posterior_mu"],
                    post_logvar=out["posterior_logvar"],
                    future_target_mask=target_mask,
                    free_nats=float(self.hparams.rssm_free_nats),
                )
                total_loss = total_loss + kl_weight * kl
                metrics["kl_loss"] = kl
                metrics["kl_weight_effective"] = torch.tensor(kl_weight, device=targets.device)

        if self.hparams.noncrossing_loss_weight > 0.0 and not self.hparams.enforce_quantile_ordering:
            nc = noncrossing_loss(preds_q, target_mask)
            total_loss = total_loss + float(self.hparams.noncrossing_loss_weight) * nc
            metrics["noncrossing_loss"] = nc

        if self.hparams.use_risk_head and "risk_logits" in out:
            risk_label = (targets < float(self.hparams.risk_threshold)).float()
            valid = ~target_mask
            if valid.any():
                risk_loss = F.binary_cross_entropy_with_logits(out["risk_logits"][valid], risk_label[valid])
            else:
                risk_loss = torch.tensor(0.0, device=targets.device)
            total_loss = total_loss + float(self.hparams.risk_loss_weight) * risk_loss
            metrics["risk_loss"] = risk_loss

        if self.hparams.compute_calibration_metrics:
            cal = calibration_metrics(preds_q, targets, target_mask, self.quantiles)
            metrics["coverage"] = cal["coverage"]
            metrics["interval_width"] = cal["interval_width"]
            metrics["calibration_error"] = cal["calibration_error"]

        metrics["latent_smoothness"] = latent_smoothness_loss(
            out["latent_states"],
            future_pad_mask=future_pad_mask,
            loss_type="l2",
        )
        if posterior_use_prob is not None and self.training:
            metrics["posterior_use_probability"] = torch.tensor(float(posterior_use_prob), device=targets.device)

        metrics["loss"] = total_loss
        return total_loss, metrics, out

    def _log_metrics(self, stage: str, metrics: dict[str, torch.Tensor], prog_bar: bool = False):
        log_dict = {f"{stage}/{k}": v for k, v in metrics.items()}
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=prog_bar, sync_dist=False)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        loss, metrics, _ = self._compute_loss_and_metrics(batch)
        self._log_metrics("train", metrics, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        loss, metrics, _ = self._compute_loss_and_metrics(batch)
        self._log_metrics("val", metrics, prog_bar=True)
        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        loss, metrics, _ = self._compute_loss_and_metrics(batch)
        self._log_metrics("test", metrics, prog_bar=False)
        return loss

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        out = self.model(batch)
        pred = {
            "quantiles": out["quantiles"],
            "latent_states": out["latent_states"],
            "future_pad_mask": batch["future_pad_mask"],
            "future_timestamps": batch.get("future_timestamps", None),
            "area": batch.get("area", None),
            "source": batch.get("source", None),
            "latitude": batch.get("latitude", None),
            "longitude": batch.get("longitude", None),
            "climate": batch.get("climate", None),
            "crop_type": batch.get("crop_type", None),
        }
        if "risk_logits" in out:
            pred["risk_logits"] = out["risk_logits"]
            pred["risk_prob"] = torch.sigmoid(out["risk_logits"])
        return pred

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
