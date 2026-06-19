#!/usr/bin/env python3
"""Scenario inference utility for the vegetation world model.

Loads a Lightning checkpoint, runs unperturbed and perturbed future-covariate
scenarios, and saves NPZ outputs for downstream analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenario import perturb_future_covariates_physical
from training.datamodule import CacheDataModule
from training.world_lightning_module import VegetationWorldLightningModule


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Scenario inference with world-model Lightning checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cache-root", type=str, required=True)
    parser.add_argument("--scaler-path", type=str, default=None, help="Explicit scaler.json path to force training scaler")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply-scaling", type=str2bool, default=True)
    parser.add_argument("--feature-engineering", type=str2bool, default=False)
    parser.add_argument("--min-crop-pixels", type=float, default=0.0)
    parser.add_argument("--data-summary-path", type=str, default="Data/dataSummary_completed.csv")
    parser.add_argument("--pin-memory", type=str2bool, default=False)
    parser.add_argument("--scenarios", type=str, default=None, help="YAML/JSON file with scenario mapping")
    parser.add_argument("--run-unperturbed-only", type=str2bool, default=False)
    return parser.parse_args()


def _default_scenarios() -> dict[str, dict]:
    return {
        "dry_20": {"precipitation_multiplier": 0.8},
        "dry_40": {"precipitation_multiplier": 0.6},
        "hot_2": {"temperature_additive": 2.0},
        "hot_4": {"temperature_additive": 4.0},
        "compound_dry40_hot4": {"precipitation_multiplier": 0.6, "temperature_additive": 4.0},
    }


def _load_scenarios(path: str | None) -> dict[str, dict]:
    if path is None:
        return _default_scenarios()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scenario config not found: {p}")
    with p.open("r", encoding="utf-8") as fp:
        if p.suffix.lower() in {".yaml", ".yml"}:
            cfg = yaml.safe_load(fp)
        else:
            cfg = json.load(fp)
    if not isinstance(cfg, dict):
        raise ValueError("Scenario config must be a mapping of scenario_name -> config")
    return cfg


def _pad_to_len(tensor: torch.Tensor, full_len: int, pad_value: float = 0.0) -> torch.Tensor:
    if tensor.shape[1] == full_len:
        return tensor
    if tensor.shape[1] > full_len:
        return tensor[:, :full_len]
    pad_shape = list(tensor.shape)
    pad_shape[1] = full_len - tensor.shape[1]
    pad = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=1)


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _pad_timestamp_rows(rows: list[list[str]], full_len: int) -> list[list[str]]:
    out: list[list[str]] = []
    for seq in rows:
        s = [str(x) for x in seq]
        if len(s) < full_len:
            s = s + [""] * (full_len - len(s))
        else:
            s = s[:full_len]
        out.append(s)
    return out


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    datamodule = CacheDataModule(
        cache_root=args.cache_root,
        scaler_path=args.scaler_path,
        predict_cache_root=args.cache_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        apply_scaling=args.apply_scaling,
        feature_engineering=args.feature_engineering,
        min_crop_pixels=args.min_crop_pixels,
        data_summary_path=args.data_summary_path,
        pin_memory=args.pin_memory,
    )
    datamodule.setup(stage="predict")
    loader = datamodule.predict_dataloader()
    if loader is None:
        raise RuntimeError("No predict dataloader available")

    feature_names = datamodule.feature_names
    if feature_names is None:
        raise RuntimeError("Could not infer feature names from cache")

    scenarios = {} if args.run_unperturbed_only else _load_scenarios(args.scenarios)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cpu")
    if torch.cuda.is_available() and args.accelerator in {"gpu", "auto"}:
        device = torch.device("cuda")

    module = VegetationWorldLightningModule.load_from_checkpoint(str(ckpt_path), map_location=device)
    module.eval()
    module.to(device)

    predict_dataset = datamodule.predict_set
    if predict_dataset is None:
        raise RuntimeError("Predict dataset missing")
    full_len = max(len(seq) for seq in predict_dataset.future_sequences)
    scenario_scaler = predict_dataset.scaler if args.apply_scaling else None

    unperturbed_quantiles = []
    unperturbed_median = []
    unperturbed_risk = []
    pad_masks = []
    latents = []

    scenario_quantiles = {name: [] for name in scenarios}
    scenario_median = {name: [] for name in scenarios}
    scenario_risk = {name: [] for name in scenarios}

    areas = []
    sources = []
    climates = []
    crop_types = []
    latitudes = []
    longitudes = []
    future_timestamps_all: list[list[str]] = []
    future_delta_days_all: list[np.ndarray] = []

    scenario_metadata: dict[str, dict] = {}

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)

            base_out = module.model(batch)
            base_q = _pad_to_len(base_out["quantiles"], full_len, pad_value=0.0)
            base_m = base_q[..., module.median_idx]

            unperturbed_quantiles.append(base_q.cpu())
            unperturbed_median.append(base_m.cpu())
            latents.append(_pad_to_len(base_out["latent_states"], full_len, pad_value=0.0).cpu())

            future_pad = _pad_to_len(batch["future_pad_mask"].unsqueeze(-1).float(), full_len, pad_value=1.0).squeeze(-1)
            pad_masks.append(future_pad.cpu().bool())
            if "future_delta_days" in batch:
                fdd = _pad_to_len(batch["future_delta_days"].unsqueeze(-1), full_len, pad_value=0.0).squeeze(-1)
                future_delta_days_all.append(fdd.cpu().numpy())

            if "risk_logits" in base_out:
                base_r = torch.sigmoid(_pad_to_len(base_out["risk_logits"], full_len, pad_value=0.0))
                unperturbed_risk.append(base_r.cpu())

            if scenarios:
                for name, cfg in scenarios.items():
                    x_pert, change = perturb_future_covariates_physical(
                        batch=batch,
                        feature_names=feature_names,
                        scenario_config=cfg,
                        scaler=scenario_scaler,
                        scenario_name=name,
                    )
                    batch_s = dict(batch)
                    batch_s["future"] = x_pert

                    out_s = module.model(batch_s)
                    s_q = _pad_to_len(out_s["quantiles"], full_len, pad_value=0.0)
                    s_m = s_q[..., module.median_idx]
                    scenario_quantiles[name].append(s_q.cpu())
                    scenario_median[name].append(s_m.cpu())

                    if "risk_logits" in out_s:
                        s_r = torch.sigmoid(_pad_to_len(out_s["risk_logits"], full_len, pad_value=0.0))
                        scenario_risk[name].append(s_r.cpu())

                    if name not in scenario_metadata:
                        scenario_metadata[name] = {
                            "precipitation_multiplier": change.precipitation_multiplier,
                            "temperature_additive": change.temperature_additive,
                            "precipitation_indices": change.precipitation_indices,
                            "temperature_indices": change.temperature_indices,
                        }

            # metadata (lists are not moved to GPU)
            areas.extend(batch["area"])
            sources.extend(batch["source"])
            climates.extend(batch["climate"])
            crop_types.extend(batch["crop_type"])
            latitudes.extend([float(v) for v in batch["latitude"]])
            longitudes.extend([float(v) for v in batch["longitude"]])
            if "future_timestamps" in batch:
                future_timestamps_all.extend(_pad_timestamp_rows(batch["future_timestamps"], full_len))

    base_q_all = torch.cat(unperturbed_quantiles, dim=0).numpy()
    base_m_all = torch.cat(unperturbed_median, dim=0).numpy()
    pad_mask_all = torch.cat(pad_masks, dim=0).numpy().astype(bool)
    latent_all = torch.cat(latents, dim=0).numpy()

    base_r_all = None
    if unperturbed_risk:
        base_r_all = torch.cat(unperturbed_risk, dim=0).numpy()

    npz_payload: dict[str, np.ndarray] = {
        "feature_names": np.array(feature_names, dtype=object),
        "quantile_levels": np.array(module.quantiles, dtype=np.float32),
        "unperturbed_quantiles": base_q_all,
        "unperturbed_median": base_m_all,
        "latent_states": latent_all,
        "future_pad_mask": pad_mask_all,
        "area": np.array(areas, dtype=object),
        "source": np.array(sources, dtype=object),
        "climate": np.array(climates, dtype=object),
        "crop_type": np.array(crop_types, dtype=object),
        "latitude": np.array(latitudes, dtype=np.float32),
        "longitude": np.array(longitudes, dtype=np.float32),
        "scenario_names": np.array(list(scenarios.keys()), dtype=object),
        "scenario_metadata_json": np.array([json.dumps(scenario_metadata)], dtype=object),
        "scaler_path": np.array([str(args.scaler_path) if args.scaler_path else ""], dtype=object),
    }
    if future_timestamps_all:
        npz_payload["future_timestamps"] = np.array(future_timestamps_all, dtype=object)
    if future_delta_days_all:
        npz_payload["future_delta_days"] = np.concatenate(future_delta_days_all, axis=0)

    if base_r_all is not None:
        npz_payload["unperturbed_risk"] = base_r_all

    for name in scenarios:
        sq = torch.cat(scenario_quantiles[name], dim=0).numpy()
        sm = torch.cat(scenario_median[name], dim=0).numpy()
        npz_payload[f"{name}__quantiles"] = sq
        npz_payload[f"{name}__median"] = sm
        npz_payload[f"{name}__delta_median"] = sm - base_m_all

        if scenario_risk[name] and base_r_all is not None:
            sr = torch.cat(scenario_risk[name], dim=0).numpy()
            npz_payload[f"{name}__risk"] = sr
            npz_payload[f"{name}__delta_risk"] = sr - base_r_all

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **npz_payload)
    print(f"Saved scenario outputs to {output_path}")


if __name__ == "__main__":
    main()
