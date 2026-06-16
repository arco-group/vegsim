#!/usr/bin/env python3
"""Evaluate baseline forecaster checkpoints with aggregate diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from world_matnet.training.datamodule import CacheDataModule
from world_matnet.training.lightning_module import QuantileLightningModule


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
    parser = argparse.ArgumentParser(description="Evaluate a baseline quantile forecaster checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cache-root", type=str, required=True)
    parser.add_argument("--scaler-path", type=str, default=None, help="Explicit scaler.json path to force training scaler")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--apply-scaling", type=str2bool, default=True)
    parser.add_argument("--feature-engineering", type=str2bool, default=False)
    parser.add_argument("--min-crop-pixels", type=float, default=0.0)
    parser.add_argument("--data-summary-path", type=str, default="Data/dataSummary_completed.csv")
    parser.add_argument("--pin-memory", type=str2bool, default=False)
    parser.add_argument(
        "--metrics-original-scale",
        type=str2bool,
        default=False,
        help="Inverse-transform predictions/targets before MAE/RMSE/pinball metrics",
    )
    return parser.parse_args()


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def _inverse_transform_target_torch(array: torch.Tensor, scaler) -> torch.Tensor:
    if scaler is None or not scaler.has_target_stats():
        return array

    mode = scaler.mode
    eps = float(scaler.eps)
    out = array
    if mode in {"standardization", "standardization_arcsinh"}:
        mean = float(np.asarray(scaler.target_mean).reshape(-1)[0])
        var = float(np.asarray(scaler.target_var).reshape(-1)[0])
        std = float(np.sqrt(var + eps))
        mean_t = torch.tensor(mean, device=array.device, dtype=array.dtype)
        std_t = torch.tensor(std, device=array.device, dtype=array.dtype)
        if mode == "standardization_arcsinh":
            out = torch.sinh(out)
        return out * std_t + mean_t
    if mode == "min-max":
        min_val = float(np.asarray(scaler.target_min_value).reshape(-1)[0])
        max_val = float(np.asarray(scaler.target_max_value).reshape(-1)[0])
        rng = float(max(max_val - min_val, eps))
        min_t = torch.tensor(min_val, device=array.device, dtype=array.dtype)
        rng_t = torch.tensor(rng, device=array.device, dtype=array.dtype)
        return out * rng_t + min_t
    raise ValueError(f"Unsupported scaler mode: {mode}")


def _ensure_4d_preds_3d_targets(
    preds_q: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if preds_q.dim() == 3:
        preds_q = preds_q.unsqueeze(2)  # [B,K,Q] -> [B,K,1,Q]
    if preds_q.dim() != 4:
        raise ValueError("Baseline predictions must be [B,K,Q] or [B,K,C,Q]")

    if targets.dim() == 2:
        targets = targets.unsqueeze(-1)  # [B,K] -> [B,K,1]
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)  # [B,K] -> [B,K,1]
    if targets.dim() != 3 or mask.dim() != 3:
        raise ValueError("Targets/masks must be [B,K] or [B,K,C]")

    if preds_q.shape[:3] != targets.shape or targets.shape != mask.shape:
        raise ValueError("Shape mismatch between preds/targets/mask")
    return preds_q, targets, mask


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    datamodule = CacheDataModule(
        cache_root=args.cache_root,
        scaler_path=args.scaler_path,
        test_cache_root=args.cache_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        apply_scaling=args.apply_scaling,
        feature_engineering=args.feature_engineering,
        min_crop_pixels=args.min_crop_pixels,
        data_summary_path=args.data_summary_path,
        pin_memory=args.pin_memory,
    )
    datamodule.setup(stage="test")
    loader = datamodule.test_dataloader()
    if loader is None:
        raise RuntimeError("No test dataloader available")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = QuantileLightningModule.load_from_checkpoint(str(ckpt_path), map_location=device)
    module.eval()
    module.to(device)

    input_dim_ckpt = int(module.hparams.input_dim)
    input_dim_data = int(datamodule.get_input_dim())
    if input_dim_ckpt != input_dim_data:
        raise RuntimeError(
            f"Input dimension mismatch: checkpoint expects {input_dim_ckpt}, dataset provides {input_dim_data}. "
            "Check --feature-engineering and cache/scaler compatibility."
        )

    eval_scaler = None
    if args.metrics_original_scale:
        eval_scaler = getattr(datamodule.test_set, "scaler", None)
        if eval_scaler is None:
            raise RuntimeError(
                "metrics_original_scale=True requires a loaded scaler. "
                "Use --apply-scaling true and/or pass --scaler-path."
            )

    quantiles = list(module.quantiles)
    q_levels = torch.tensor(quantiles, device=device, dtype=torch.float32).view(1, 1, 1, -1)
    median_idx = int(module.median_idx)

    pin_num = 0.0
    pin_den = 0.0
    mae_num = 0.0
    mse_num = 0.0
    reg_den = 0.0

    coverage_num = 0.0
    width_num = 0.0
    cal_error_acc = np.zeros(len(quantiles), dtype=np.float64)
    cal_den = 0.0

    per_rank = defaultdict(lambda: {"mae": 0.0, "mse": 0.0, "pin": 0.0, "den": 0.0, "wden": 0.0})
    per_horizon_days = defaultdict(lambda: {"mae": 0.0, "mse": 0.0, "pin": 0.0, "den": 0.0, "wden": 0.0})
    per_region = defaultdict(lambda: {"mae": 0.0, "mse": 0.0, "pin": 0.0, "den": 0.0, "wden": 0.0})

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            preds_q = module.model(batch)
            targets = batch["target"]
            mask = batch["target_mask"]
            preds_q, targets, mask = _ensure_4d_preds_3d_targets(preds_q, targets, mask)

            if args.metrics_original_scale:
                preds_metric = _inverse_transform_target_torch(preds_q, eval_scaler)
                targets_metric = _inverse_transform_target_torch(targets, eval_scaler)
            else:
                preds_metric = preds_q
                targets_metric = targets

            keep = (~mask).float()
            if keep.sum() == 0:
                continue

            weights = keep
            delta_days = batch.get("target_delta_days", None)
            if module.hparams.time_weight_alpha is not None and delta_days is not None:
                if delta_days.dim() == 2:
                    delta_days = delta_days.unsqueeze(-1)
                w_time = 1.0 / (1.0 + float(module.hparams.time_weight_alpha) * torch.clamp(delta_days, min=0.0))
                weights = keep * w_time

            diff_q = targets_metric.unsqueeze(-1) - preds_metric
            pin = torch.maximum(q_levels * diff_q, (q_levels - 1.0) * diff_q)
            pin_num += float((pin * weights.unsqueeze(-1)).sum().item())
            pin_den += float(weights.sum().item()) * len(quantiles)

            median = preds_metric[..., median_idx]
            abs_diff = (median - targets_metric).abs() * keep
            sq_diff = (median - targets_metric).pow(2) * keep
            mae_num += float(abs_diff.sum().item())
            mse_num += float(sq_diff.sum().item())
            reg_den += float(keep.sum().item())

            if preds_metric.shape[-1] >= 2:
                low = preds_metric[..., 0]
                high = preds_metric[..., -1]
                inside = ((targets_metric >= low) & (targets_metric <= high) & (~mask)).float()
                coverage_num += float(inside.sum().item())
                width_num += float(((high - low).abs() * keep).sum().item())

            for qi, tau in enumerate(quantiles):
                event = ((targets_metric <= preds_metric[..., qi]) & (~mask)).float()
                cal_error_acc[qi] += float(event.sum().item())
            cal_den += float(keep.sum().item())

            bsz, seq_len, channels = targets_metric.shape
            for k in range(seq_len):
                keep_k = keep[:, k, :]
                den_k = float(keep_k.sum().item())
                if den_k == 0.0:
                    continue
                abs_k = ((median[:, k, :] - targets_metric[:, k, :]).abs() * keep_k).sum().item()
                sq_k = (((median[:, k, :] - targets_metric[:, k, :]) ** 2) * keep_k).sum().item()
                pin_k = torch.maximum(
                    q_levels * diff_q[:, k : k + 1, :, :],
                    (q_levels - 1.0) * diff_q[:, k : k + 1, :, :],
                )
                w_k = weights[:, k : k + 1, :].unsqueeze(-1)
                pin_k_num = float((pin_k * w_k).sum().item())
                wden_k = float(weights[:, k : k + 1, :].sum().item())
                per_rank[k]["mae"] += abs_k
                per_rank[k]["mse"] += sq_k
                per_rank[k]["pin"] += pin_k_num
                per_rank[k]["den"] += den_k
                per_rank[k]["wden"] += wden_k

            climates = batch["climate"]
            if delta_days is not None:
                delta_days_int = torch.round(delta_days).long()
            else:
                delta_days_int = None

            for b in range(bsz):
                reg = str(climates[b])
                keep_b = keep[b]
                den_b = float(keep_b.sum().item())
                if den_b == 0.0:
                    continue

                abs_b = ((median[b] - targets_metric[b]).abs() * keep_b).sum().item()
                sq_b = (((median[b] - targets_metric[b]) ** 2) * keep_b).sum().item()
                pin_b = torch.maximum(q_levels * diff_q[b : b + 1], (q_levels - 1.0) * diff_q[b : b + 1])
                w_b = weights[b : b + 1].unsqueeze(-1)
                pin_b_num = float((pin_b * w_b).sum().item())
                wden_b = float(weights[b : b + 1].sum().item())
                per_region[reg]["mae"] += abs_b
                per_region[reg]["mse"] += sq_b
                per_region[reg]["pin"] += pin_b_num
                per_region[reg]["den"] += den_b
                per_region[reg]["wden"] += wden_b

                if delta_days_int is None:
                    continue
                for k in range(seq_len):
                    for c in range(channels):
                        if not bool((~mask[b, k, c]).item()):
                            continue
                        day = int(delta_days_int[b, k, c].item())
                        w_v = float(weights[b, k, c].item())
                        abs_v = float((median[b, k, c] - targets_metric[b, k, c]).abs().item())
                        sq_v = float(((median[b, k, c] - targets_metric[b, k, c]) ** 2).item())
                        pin_v = float(pin[b, k, c].sum().item()) * w_v
                        per_horizon_days[day]["mae"] += abs_v
                        per_horizon_days[day]["mse"] += sq_v
                        per_horizon_days[day]["pin"] += pin_v
                        per_horizon_days[day]["den"] += 1.0
                        per_horizon_days[day]["wden"] += w_v

    overall_pin = _safe_div(pin_num, pin_den)
    overall_mae = _safe_div(mae_num, reg_den)
    overall_rmse = np.sqrt(_safe_div(mse_num, reg_den))
    coverage = _safe_div(coverage_num, reg_den)
    interval_width = _safe_div(width_num, reg_den)

    cal_errors = []
    if cal_den > 0:
        for qi, tau in enumerate(quantiles):
            empirical = cal_error_acc[qi] / cal_den
            cal_errors.append(abs(empirical - float(tau)))
    calibration_error = float(np.mean(cal_errors)) if cal_errors else 0.0

    per_rank_out = {}
    for rank, values in sorted(per_rank.items(), key=lambda kv: kv[0]):
        den = values["den"]
        per_rank_out[int(rank)] = {
            "mae": _safe_div(values["mae"], den),
            "rmse": float(np.sqrt(_safe_div(values["mse"], den))),
            "weighted_pinball": _safe_div(values["pin"], values["wden"] * len(quantiles)),
        }

    per_horizon_days_out = {}
    for day, values in sorted(per_horizon_days.items(), key=lambda kv: kv[0]):
        den = values["den"]
        per_horizon_days_out[int(day)] = {
            "mae": _safe_div(values["mae"], den),
            "rmse": float(np.sqrt(_safe_div(values["mse"], den))),
            "weighted_pinball": _safe_div(values["pin"], values["wden"] * len(quantiles)),
        }

    per_region_out = {}
    for region, values in sorted(per_region.items(), key=lambda kv: kv[0]):
        den = values["den"]
        per_region_out[region] = {
            "mae": _safe_div(values["mae"], den),
            "rmse": float(np.sqrt(_safe_div(values["mse"], den))),
            "weighted_pinball": _safe_div(values["pin"], values["wden"] * len(quantiles)),
        }

    result = {
        "overall": {
            "mae": overall_mae,
            "rmse": float(overall_rmse),
            "weighted_pinball": overall_pin,
            "coverage": coverage,
            "interval_width": interval_width,
            "calibration_error": calibration_error,
        },
        "per_target_rank": per_rank_out,
        "per_horizon_days": per_horizon_days_out,
        "per_region": per_region_out,
        "quantiles": quantiles,
        "metrics_original_scale": bool(args.metrics_original_scale),
        "scaler_path": str(args.scaler_path) if args.scaler_path else None,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)

    print(f"Saved baseline evaluation metrics to {output_path}")


if __name__ == "__main__":
    main()
