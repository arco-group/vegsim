"""CLI training entrypoint using PyTorch Lightning.

Supports:
- baseline_forecaster (existing quantile model)
- vegetation_world_model (new modular world model)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    try:
        from lightning.pytorch.loggers import WandbLogger
    except ImportError:  # pragma: no cover
        WandbLogger = None
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    try:
        from pytorch_lightning.loggers import WandbLogger
    except ImportError:  # pragma: no cover
        WandbLogger = None

import torch
import yaml

from world_matnet.training.batch import build_feature_index
from world_matnet.training.datamodule import CacheDataModule
from world_matnet.training.lightning_module import QuantileLightningModule
from world_matnet.training.utils import set_seed, str2bool
from world_matnet.training.world_lightning_module import VegetationWorldLightningModule


def parse_quantiles(value):
    if isinstance(value, list):
        quantiles = [float(v) for v in value]
    else:
        parts = str(value).split(",")
        quantiles = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            try:
                tau = float(part)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"Invalid quantile value: {part}") from exc
            quantiles.append(tau)

    if not quantiles:
        raise argparse.ArgumentTypeError("At least one quantile is required")
    quantiles_sorted = sorted(quantiles)
    if quantiles_sorted != quantiles:
        raise argparse.ArgumentTypeError("Quantiles must be in increasing order")
    if len(set(quantiles)) != len(quantiles):
        raise argparse.ArgumentTypeError("Quantiles must be unique")
    for tau in quantiles:
        if not (0.0 < tau < 1.0):
            raise argparse.ArgumentTypeError(f"Quantile {tau} must be in (0,1)")
    return quantiles


def parse_devices(value):
    if value is None:
        return "auto"
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    if "," in text:
        return [int(x.strip()) for x in text.split(",") if x.strip()]
    try:
        return int(text)
    except ValueError:
        return text


def parse_int_list(value: str | list[int] | None, default: list[int] | None = None) -> list[int] | None:
    if value is None:
        if default is None:
            return None
        return list(default)
    if isinstance(value, list):
        return [int(v) for v in value]
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return [int(p) for p in parts]


def parse_str_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [p.strip() for p in str(value).split(",") if p.strip()]


def _load_yaml_config(path: str | None) -> dict:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must define a mapping")
    return data


def build_parser(config_defaults: dict | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vegetation forecasting/world-model Lightning training")
    parser.add_argument("--config", type=str, default=None, help="YAML config preset")

    # Data and training setup
    parser.add_argument("--cache-root", default="Data/cache/train_avg_NDVI_clear_sky")
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default="checkpoints",
        help="Root directory where run checkpoints/metadata are saved",
    )
    parser.add_argument("--scaler-path", type=str, default=None)
    parser.add_argument("--val-cache-root", type=str, default=None)
    parser.add_argument("--test-cache-root", type=str, default=None)
    parser.add_argument("--predict-cache-root", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None, help="Path to a Lightning checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--saving-frequency", type=int, default=1)
    parser.add_argument("--min-crop-pixels", type=float, default=0.0)
    parser.add_argument("--data-summary-path", type=str, default="Data/dataSummary_completed.csv")
    parser.add_argument("--quantiles", type=parse_quantiles, default=[0.1, 0.5, 0.9])
    parser.add_argument("--apply-scaling", type=str2bool, default=True)
    parser.add_argument("--feature-engineering", type=str2bool, default=False)
    parser.add_argument("--discretize-target", type=str2bool, default=False)
    parser.add_argument("--experiment-name", type=str, default="quantile_lightning")
    parser.add_argument("--model-type", type=str, default="baseline_forecaster", choices=["baseline_forecaster", "vegetation_world_model"])

    # LR scheduling
    parser.add_argument("--lr-reduce-on-plateau", type=str2bool, default=False)
    parser.add_argument("--lr-factor", type=float, default=0.2)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--lr-min", type=float, default=5e-5)

    # Shared encoder/backbone
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Baseline ablations
    parser.add_argument("--time-weight-alpha", type=float, default=None)
    parser.add_argument("--ablate-future-covariates", type=str2bool, default=False)
    parser.add_argument("--ablate-history-covariates", type=str2bool, default=False)
    parser.add_argument("--ablate-target-history", type=str2bool, default=False)

    # World-model architecture
    parser.add_argument("--target-dim", type=int, default=1)
    parser.add_argument("--summarizer-type", type=str, default="attention_pooling", choices=["attention_pooling", "posterior_attention_pooling", "masked_average_pooling"])
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--sample-initial-state", type=str2bool, default=True)
    parser.add_argument("--initial-state-kl-weight", type=float, default=0.0)

    parser.add_argument("--use-horizon-embedding", type=str2bool, default=True)
    parser.add_argument("--horizon-emb-dim", type=int, default=32)
    parser.add_argument("--horizon-embedding-type", type=str, default="learned", choices=["learned", "sinusoidal", "mlp"])
    parser.add_argument("--max-horizon", type=int, default=512)

    parser.add_argument("--spatial-enabled", type=str2bool, default=False)
    parser.add_argument("--spatial-cont-dim", type=int, default=2)
    parser.add_argument(
        "--spatial-cat-cardinalities",
        type=str,
        default=None,
        help="Comma-separated cardinalities for categorical spatial fields. "
        "Use an empty value to disable categorical spatial metadata.",
    )
    parser.add_argument("--spatial-emb-dim", type=int, default=32)
    parser.add_argument("--use-harmonic-coords", type=str2bool, default=False)
    parser.add_argument("--num-harmonic-frequencies", type=int, default=4)
    parser.add_argument("--spatial-dropout", type=float, default=0.1)

    parser.add_argument(
        "--dynamics-type",
        type=str,
        default="gru",
        choices=["residual_mlp", "gru", "transformer", "rssm"],
    )
    parser.add_argument("--dynamics-hidden-dim", type=int, default=128)
    parser.add_argument("--dynamics-num-layers", type=int, default=2)
    parser.add_argument("--use-residual-update", type=str2bool, default=False)
    parser.add_argument("--sample-latent", type=str2bool, default=True)

    parser.add_argument("--rssm-deterministic-dim", type=int, default=128)
    parser.add_argument("--rssm-stochastic-dim", type=int, default=128)
    parser.add_argument("--rssm-hidden-dim", type=int, default=128)

    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-num-layers", type=int, default=2)
    parser.add_argument("--use-horizon-conditioned-decoder", type=str2bool, default=False)
    parser.add_argument("--enforce-quantile-ordering", type=str2bool, default=False)

    parser.add_argument("--use-risk-head", type=str2bool, default=False)
    parser.add_argument("--risk-threshold", type=float, default=0.2)

    # World-model losses
    parser.add_argument("--smoothness-loss-weight", type=float, default=0.0)
    parser.add_argument("--smoothness-loss-type", type=str, default="l2", choices=["l2", "huber"])
    parser.add_argument("--aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--aux-loss-type", type=str, default="mae", choices=["mae", "mse", "huber"])
    parser.add_argument("--use-aux-decoder", type=str2bool, default=False)
    parser.add_argument("--rssm-kl-weight", type=float, default=0.0)
    parser.add_argument("--rssm-free-nats", type=float, default=0.0)
    parser.add_argument("--rssm-kl-warmup-epochs", type=int, default=0)
    parser.add_argument("--posterior-mixing-enabled", type=str2bool, default=False)
    parser.add_argument("--posterior-mixing-start-prob", type=float, default=1.0)
    parser.add_argument("--posterior-mixing-end-prob", type=float, default=1.0)
    parser.add_argument("--posterior-mixing-anneal-epochs", type=int, default=0)
    parser.add_argument("--noncrossing-loss-weight", type=float, default=0.0)
    parser.add_argument("--sigreg-weight", type=float, default=0.0)
    parser.add_argument("--sigreg-num-proj", type=int, default=256)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--use-sigreg-projector", type=str2bool, default=False)
    parser.add_argument("--sigreg-projector-hidden-dim", type=int, default=128)
    parser.add_argument("--sigreg-projector-output-dim", type=int, default=None)
    parser.add_argument("--sigreg-projector-num-layers", type=int, default=1)
    parser.add_argument("--sigreg-projector-dropout", type=float, default=0.0)
    parser.add_argument("--risk-loss-weight", type=float, default=0.0)
    parser.add_argument("--compute-calibration-metrics", type=str2bool, default=False)

    # Trainer
    parser.add_argument("--accelerator", type=str, default="auto", help="cpu|gpu|auto")
    parser.add_argument("--devices", type=str, default="auto", help="auto|1|0,1")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--fast-dev-run", type=str2bool, default=False)
    parser.add_argument("--pin-memory", type=str2bool, default=False)

    # Evaluation/checkpointing
    parser.add_argument("--run-test-after-fit", type=str2bool, default=False)
    parser.add_argument("--checkpoint-monitor", type=str, default=None)
    parser.add_argument("--checkpoint-mode", type=str, default="min", choices=["min", "max"])

    # Logging backends
    parser.add_argument("--logger-type", type=str, default="wandb", choices=["csv", "wandb", "both"])
    parser.add_argument("--wandb-project", type=str, default="vegsim")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated tags")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-job-type", type=str, default="train")
    parser.add_argument("--wandb-offline", type=str2bool, default=False)
    parser.add_argument("--wandb-log-model", type=str2bool, default=False)

    if config_defaults:
        parser.set_defaults(**config_defaults)

    return parser


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    config_defaults = _load_yaml_config(pre_args.config)
    parser = build_parser(config_defaults=config_defaults)
    args = parser.parse_args()

    args.quantiles = parse_quantiles(args.quantiles)
    args.spatial_cat_cardinalities = parse_int_list(args.spatial_cat_cardinalities, default=None)
    args.wandb_tags = parse_str_list(args.wandb_tags)
    return args


def _build_model(args, datamodule):
    feature_names = datamodule.feature_names
    input_dim = datamodule.get_input_dim()
    target_dim_data = datamodule.get_target_dim()
    target_dim = max(1, int(args.target_dim or target_dim_data))

    if args.model_type == "baseline_forecaster":
        feature_idx = build_feature_index(feature_names)
        ablation_cfg = {
            "future_covariates_off": args.ablate_future_covariates,
            "history_covariates_off": args.ablate_history_covariates,
            "target_history_off": args.ablate_target_history,
        }
        model = QuantileLightningModule(
            input_dim=input_dim,
            quantiles=args.quantiles,
            lr=args.lr,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            ablation_cfg=ablation_cfg,
            feature_idx=feature_idx,
            time_weight_alpha=args.time_weight_alpha,
            lr_reduce_on_plateau=args.lr_reduce_on_plateau,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            lr_min=args.lr_min,
        )
        return model

    spatial_cards = args.spatial_cat_cardinalities
    if spatial_cards is None:
        spatial_cards = datamodule.get_spatial_cat_cardinalities()

    model = VegetationWorldLightningModule(
        input_dim=input_dim,
        target_dim=target_dim,
        quantiles=args.quantiles,
        lr=args.lr,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        summarizer_type=args.summarizer_type,
        latent_dim=args.latent_dim,
        sample_initial_state=args.sample_initial_state,
        initial_state_kl_weight=args.initial_state_kl_weight,
        use_horizon_embedding=args.use_horizon_embedding,
        horizon_emb_dim=args.horizon_emb_dim,
        horizon_embedding_type=args.horizon_embedding_type,
        max_horizon=args.max_horizon,
        spatial_enabled=args.spatial_enabled,
        spatial_cont_dim=args.spatial_cont_dim,
        spatial_cat_cardinalities=spatial_cards,
        spatial_emb_dim=args.spatial_emb_dim,
        use_harmonic_coords=args.use_harmonic_coords,
        num_harmonic_frequencies=args.num_harmonic_frequencies,
        spatial_dropout=args.spatial_dropout,
        dynamics_type=args.dynamics_type,
        dynamics_hidden_dim=args.dynamics_hidden_dim,
        dynamics_num_layers=args.dynamics_num_layers,
        use_residual_update=args.use_residual_update,
        rssm_deterministic_dim=args.rssm_deterministic_dim,
        rssm_stochastic_dim=args.rssm_stochastic_dim,
        rssm_hidden_dim=args.rssm_hidden_dim,
        sample_latent=args.sample_latent,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_num_layers=args.decoder_num_layers,
        use_horizon_conditioned_decoder=args.use_horizon_conditioned_decoder,
        enforce_quantile_ordering=args.enforce_quantile_ordering,
        use_risk_head=args.use_risk_head,
        risk_threshold=args.risk_threshold,
        risk_loss_weight=args.risk_loss_weight,
        smoothness_loss_weight=args.smoothness_loss_weight,
        smoothness_loss_type=args.smoothness_loss_type,
        aux_loss_weight=args.aux_loss_weight,
        aux_loss_type=args.aux_loss_type,
        use_aux_decoder=args.use_aux_decoder,
        rssm_kl_weight=args.rssm_kl_weight,
        rssm_free_nats=args.rssm_free_nats,
        rssm_kl_warmup_epochs=args.rssm_kl_warmup_epochs,
        posterior_mixing_enabled=args.posterior_mixing_enabled,
        posterior_mixing_start_prob=args.posterior_mixing_start_prob,
        posterior_mixing_end_prob=args.posterior_mixing_end_prob,
        posterior_mixing_anneal_epochs=args.posterior_mixing_anneal_epochs,
        noncrossing_loss_weight=args.noncrossing_loss_weight,
        sigreg_weight=args.sigreg_weight,
        sigreg_num_proj=args.sigreg_num_proj,
        sigreg_knots=args.sigreg_knots,
        use_sigreg_projector=args.use_sigreg_projector,
        sigreg_projector_hidden_dim=args.sigreg_projector_hidden_dim,
        sigreg_projector_output_dim=args.sigreg_projector_output_dim,
        sigreg_projector_num_layers=args.sigreg_projector_num_layers,
        sigreg_projector_dropout=args.sigreg_projector_dropout,
        compute_calibration_metrics=args.compute_calibration_metrics,
        time_weight_alpha=args.time_weight_alpha,
        lr_reduce_on_plateau=args.lr_reduce_on_plateau,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_min=args.lr_min,
    )
    return model


def _default_monitor(args) -> str:
    if args.checkpoint_monitor:
        return args.checkpoint_monitor
    return "val/loss"


def _model_name_tag(model_type: str) -> str:
    if model_type == "vegetation_world_model":
        return "wm"
    if model_type == "baseline_forecaster":
        return "baseline"
    return str(model_type)


def _save_run_metadata(output_dir: Path, args, datamodule, monitor: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = vars(args).copy()
    meta["timestamp"] = datetime.utcnow().isoformat() + "Z"
    meta["input_dim"] = datamodule.get_input_dim()
    meta["target_dim_detected"] = datamodule.get_target_dim()
    meta["feature_names"] = datamodule.feature_names
    meta["spatial_cat_cardinalities_detected"] = datamodule.get_spatial_cat_cardinalities()
    meta["checkpoint_monitor"] = monitor

    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as fp:
        yaml.safe_dump(meta, fp, sort_keys=False)


def _build_loggers(args, output_dir: Path, run_name: str):
    loggers = []
    logger_type = args.logger_type

    if logger_type in {"csv", "both"}:
        loggers.append(CSVLogger(save_dir=str(output_dir), name="logs"))

    if logger_type in {"wandb", "both"}:
        if WandbLogger is None:
            raise ImportError(
                "WandbLogger requested but unavailable. Install wandb and pytorch-lightning/lightning extras."
            )
        wandb_name = args.wandb_run_name or run_name
        try:
            loggers.append(
                WandbLogger(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=wandb_name,
                    save_dir=str(output_dir),
                    tags=args.wandb_tags,
                    group=args.wandb_group,
                    job_type=args.wandb_job_type,
                    offline=args.wandb_offline,
                    log_model=args.wandb_log_model,
                )
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "W&B logging requested but `wandb` is not installed. Run: pip install -U 'wandb>=0.12.10'"
            ) from exc

    if not loggers:
        return False
    if len(loggers) == 1:
        return loggers[0]
    return loggers


def main():
    args = parse_args()

    print("===== Lightning Training Configuration =====")
    print(json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}, indent=2))
    print("============================================")

    set_seed(args.seed)

    datamodule = CacheDataModule(
        cache_root=args.cache_root,
        scaler_path=args.scaler_path,
        val_cache_root=args.val_cache_root,
        test_cache_root=args.test_cache_root,
        predict_cache_root=args.predict_cache_root,
        batch_size=args.batch_size,
        train_split=args.train_split,
        seed=args.seed,
        num_workers=args.num_workers,
        apply_scaling=args.apply_scaling,
        data_summary_path=args.data_summary_path,
        min_crop_pixels=args.min_crop_pixels,
        feature_engineering=args.feature_engineering,
        discretize_target=args.discretize_target,
        pin_memory=args.pin_memory,
    )
    datamodule.setup()

    model = _build_model(args, datamodule)

    run_name = args.experiment_name
    model_tag = _model_name_tag(args.model_type)
    if args.model_type == "vegetation_world_model":
        run_name = f"{run_name}_{model_tag}_{args.dynamics_type}_seed{args.seed}"

    output_dir = Path(args.checkpoint_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    monitor_metric = _default_monitor(args)
    _save_run_metadata(output_dir, args, datamodule, monitor=monitor_metric)

    checkpoint_best = ModelCheckpoint(
        dirpath=output_dir,
        filename=f"best-{model_tag}-{args.dynamics_type}" + "-{epoch:02d}-{step}",
        monitor=monitor_metric,
        mode=args.checkpoint_mode,
        save_top_k=1,
    )
    checkpoint_last = ModelCheckpoint(
        dirpath=output_dir,
        filename=f"last-{model_tag}-{args.dynamics_type}" + "-{epoch:02d}",
        save_top_k=1,
        save_last=True,
        every_n_epochs=max(1, int(args.saving_frequency)),
    )

    logger = _build_loggers(args, output_dir=output_dir, run_name=run_name)

    devices = parse_devices(args.devices)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=devices,
        precision=args.precision,
        callbacks=[checkpoint_best, checkpoint_last],
        logger=logger,
        default_root_dir=str(output_dir),
        fast_dev_run=args.fast_dev_run,
        deterministic=True,
        log_every_n_steps=10,
    )

    ckpt_path = None
    if args.resume_from:
        path = Path(args.resume_from)
        if path.exists():
            ckpt_path = str(path)
        else:
            print(f"Warning: resume checkpoint not found: {path}")

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    if args.run_test_after_fit and datamodule.test_dataloader() is not None:
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    final_weights = output_dir / "model_final.pth"
    torch.save(model.model.state_dict(), final_weights)
    print(f"Saved final weights to {final_weights}")


if __name__ == "__main__":
    main()
