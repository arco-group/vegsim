#!/usr/bin/env python3
"""Evaluate world-model checkpoints with aggregate, per-horizon, and per-region metrics."""

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
    parser = argparse.ArgumentParser(description="Evaluate a vegetation world-model checkpoint")
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
    parser.add_argument(
        "--metrics-style",
        type=str,
        default="current",
        choices=["current", "agrimatnet_v2"],
        help="Metric computation style: current pipeline or AgrimatNet-like v2 for comparability.",
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
        out = out * std_t + mean_t
        return out
    if mode == "min-max":
        min_val = float(np.asarray(scaler.target_min_value).reshape(-1)[0])
        max_val = float(np.asarray(scaler.target_max_value).reshape(-1)[0])
        rng = float(max(max_val - min_val, eps))
        min_t = torch.tensor(min_val, device=array.device, dtype=array.dtype)
        rng_t = torch.tensor(rng, device=array.device, dtype=array.dtype)
        return out * rng_t + min_t
    raise ValueError(f"Unsupported scaler mode: {mode}")


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
    module = VegetationWorldLightningModule.load_from_checkpoint(str(ckpt_path), map_location=device)
    module.eval()
    module.to(device)

    eval_scaler = None
    if args.metrics_original_scale:
        eval_scaler = getattr(datamodule.test_set, "scaler", None)
        if eval_scaler is None:
            raise RuntimeError(
                "metrics_original_scale=True requires a loaded scaler. "
                "Use --apply-scaling true and/or pass --scaler-path."
            )

    quantiles = module.quantiles
    q_levels = torch.tensor(quantiles, device=device, dtype=torch.float32).view(1, 1, 1, -1)
    median_idx = module.median_idx

    pin_num = 0.0
    pin_den = 0.0
    mae_num = 0.0
    mse_num = 0.0
    reg_den = 0.0

    coverage_num = 0.0
    width_num = 0.0
    cal_error_acc = np.zeros(len(quantiles), dtype=np.float64)
    cal_den = 0.0

    per_h_num = defaultdict(lambda: {"mae": 0.0, "mse": 0.0, "pin": 0.0, "den": 0.0, "wden": 0.0})
    per_region = defaultdict(lambda: {"mae": 0.0, "mse": 0.0, "pin": 0.0, "den": 0.0, "wden": 0.0})
    mae_per_sample: list[float] = []
    rmse_per_sample: list[float] = []
    pinball_per_sample: list[float] = []
    wmape_per_sample: list[float] = []
    crps_per_sample: list[float] = []
    total_abs_error_sq_items = 0.0
    total_abs_error_item_count = 0
    total_naive_abs_error = 0.0
    total_naive_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            out = module.model(batch)

            preds_q = out["quantiles"]  # [B,L,C,Q]
            targets = batch["target_dense"]
            mask = batch["target_dense_mask"]

            if args.metrics_original_scale:
                preds_metric = _inverse_transform_target_torch(preds_q, eval_scaler)
                targets_metric = _inverse_transform_target_torch(targets, eval_scaler)
            else:
                preds_metric = preds_q
                targets_metric = targets

            keep = (~mask).float()
            if keep.sum() == 0:
                continue

            use_time_weights = args.metrics_style != "agrimatnet_v2"
            weights = keep
            if use_time_weights and module.hparams.time_weight_alpha is not None:
                delta = torch.clamp(batch["future_delta_days"], min=0.0)
                w_time = 1.0 / (1.0 + float(module.hparams.time_weight_alpha) * delta)
                weights = keep * w_time.unsqueeze(-1)

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

            bsz, seq_len, _ = targets.shape
            for h in range(seq_len):
                keep_h = keep[:, h, :]
                den_h = float(keep_h.sum().item())
                if den_h == 0.0:
                    continue

                abs_h = ((median[:, h, :] - targets_metric[:, h, :]).abs() * keep_h).sum().item()
                sq_h = (((median[:, h, :] - targets_metric[:, h, :]) ** 2) * keep_h).sum().item()

                pin_h = torch.maximum(
                    q_levels[:, :, :, :] * diff_q[:, h : h + 1, :, :],
                    (q_levels[:, :, :, :] - 1.0) * diff_q[:, h : h + 1, :, :],
                )
                weight_h = weights[:, h : h + 1, :].unsqueeze(-1)
                pin_h_num = float((pin_h * weight_h).sum().item())
                wden_h = float(weights[:, h : h + 1, :].sum().item())

                per_h_num[h]["mae"] += abs_h
                per_h_num[h]["mse"] += sq_h
                per_h_num[h]["pin"] += pin_h_num
                per_h_num[h]["den"] += den_h
                per_h_num[h]["wden"] += wden_h

            climates = batch["climate"]
            for i, region in enumerate(climates):
                keep_i = keep[i]
                den_i = float(keep_i.sum().item())
                if den_i == 0.0:
                    continue

                abs_i = ((median[i] - targets_metric[i]).abs() * keep_i).sum().item()
                sq_i = (((median[i] - targets_metric[i]) ** 2) * keep_i).sum().item()

                pin_i = torch.maximum(
                    q_levels * diff_q[i : i + 1],
                    (q_levels - 1.0) * diff_q[i : i + 1],
                )
                w_i = weights[i : i + 1].unsqueeze(-1)
                pin_i_num = float((pin_i * w_i).sum().item())
                wden_i = float(weights[i : i + 1].sum().item())

                reg = str(region)
                per_region[reg]["mae"] += abs_i
                per_region[reg]["mse"] += sq_i
                per_region[reg]["pin"] += pin_i_num
                per_region[reg]["den"] += den_i
                per_region[reg]["wden"] += wden_i

            # Additional sample-wise metrics (mean/std over samples)
            bsz, seq_len, _ = targets_metric.shape
            y_hist_batch = batch.get("y_hist")
            hist_mask_batch = batch.get("hist_mask")
            if isinstance(y_hist_batch, torch.Tensor) and args.metrics_original_scale:
                y_hist_metric_batch = _inverse_transform_target_torch(y_hist_batch, eval_scaler)
            else:
                y_hist_metric_batch = y_hist_batch
            for i in range(bsz):
                keep_i = keep[i]
                den_i = float(keep_i.sum().item())
                if den_i == 0.0:
                    continue

                median_i = median[i]
                target_i = targets_metric[i]
                abs_err_i = (median_i - target_i).abs() * keep_i
                sq_err_i = (median_i - target_i).pow(2) * keep_i

                mae_per_sample.append(float(abs_err_i.sum().item() / den_i))
                rmse_per_sample.append(float(np.sqrt(sq_err_i.sum().item() / den_i)))

                target_abs_sum = float((target_i.abs() * keep_i).sum().item())
                if target_abs_sum > 0.0:
                    wmape_per_sample.append(float(abs_err_i.sum().item() / target_abs_sum))

                valid_i = keep_i > 0.5
                if valid_i.any():
                    abs_vals_i = (median_i - target_i).abs()[valid_i]
                    total_abs_error_sq_items += float((abs_vals_i.pow(2)).sum().item())
                    total_abs_error_item_count += int(abs_vals_i.numel())

                # MASE variant: naive reference = last valid historical target value.
                if isinstance(y_hist_metric_batch, torch.Tensor):
                    y_hist_i = y_hist_metric_batch[i]
                    if y_hist_i.dim() > 1:
                        y_hist_i = y_hist_i[..., 0]
                    if isinstance(hist_mask_batch, torch.Tensor):
                        hm = hist_mask_batch[i]
                        if hm.dim() > 1:
                            hm = hm[..., 0]
                        valid_hist = ~hm.bool()
                    else:
                        valid_hist = torch.ones(y_hist_i.shape[0], device=y_hist_i.device, dtype=torch.bool)

                    hist_valid = y_hist_i[valid_hist]
                    if hist_valid.numel() > 0 and valid_i.any():
                        naive_val = hist_valid[-1]
                        target_vals = target_i[valid_i]
                        naive_abs = (target_vals - naive_val).abs()
                        total_naive_abs_error += float(naive_abs.sum().item())
                        total_naive_count += int(target_vals.numel())

                # CRPS:
                # - current: 2 * mean_tau pinball_tau
                # - agrimatnet_v2: 2 * trapz(pinball_tau over quantiles)
                diff_q_i = target_i.unsqueeze(-1) - preds_metric[i]
                pin_i = torch.maximum(q_levels * diff_q_i.unsqueeze(0), (q_levels - 1.0) * diff_q_i.unsqueeze(0))
                # pin_i shape [1, L, C, Q]
                keep_i_q = keep_i.unsqueeze(-1)
                den_q = float(keep_i_q.sum().item()) * len(quantiles)
                if den_q > 0.0:
                    mean_pin_i = float((pin_i.squeeze(0) * keep_i_q).sum().item() / den_q)
                    pinball_per_sample.append(mean_pin_i)
                    if args.metrics_style == "agrimatnet_v2":
                        # Align mask shape to [L, C, Q] before boolean indexing.
                        losses_np = (pin_i.squeeze(0) * keep_i_q).detach().cpu().numpy()  # [L,C,Q]
                        valid_np = keep_i_q.expand_as(pin_i.squeeze(0)).detach().cpu().numpy()  # [L,C,Q]
                        vals = losses_np[valid_np > 0.5]
                        if vals.size > 0:
                            vals = vals.reshape(-1, len(quantiles))
                            crps_i = 2.0 * float(np.trapz(vals, np.asarray(quantiles, dtype=np.float64), axis=-1).mean())
                            crps_per_sample.append(crps_i)
                        else:
                            crps_per_sample.append(0.0)
                    else:
                        crps_per_sample.append(2.0 * mean_pin_i)

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
    mae_mean = float(np.mean(mae_per_sample)) if mae_per_sample else overall_mae
    mae_std = float(np.std(mae_per_sample)) if mae_per_sample else 0.0
    rmse_mean = float(np.mean(rmse_per_sample)) if rmse_per_sample else float(overall_rmse)
    rmse_std = float(np.std(rmse_per_sample)) if rmse_per_sample else 0.0
    pinball_mean = float(np.mean(pinball_per_sample)) if pinball_per_sample else overall_pin
    pinball_std = float(np.std(pinball_per_sample)) if pinball_per_sample else 0.0
    wmape_mean = float(np.mean(wmape_per_sample)) if wmape_per_sample else 0.0
    wmape_std = float(np.std(wmape_per_sample)) if wmape_per_sample else 0.0
    if total_naive_count > 0 and total_naive_abs_error > 0.0 and reg_den > 0.0:
        naive_denom = total_naive_abs_error / float(total_naive_count)
        mase_mean = overall_mae / naive_denom
        if args.metrics_style == "agrimatnet_v2":
            # Match AgrimatNet script: mase_std from mae_std / naive_denom
            mase_std = mae_std / naive_denom if naive_denom > 0.0 else float("nan")
        elif total_abs_error_item_count > 1:
            mae_item_mean = overall_mae
            mae_item_var = (total_abs_error_sq_items / float(total_abs_error_item_count)) - (mae_item_mean ** 2)
            mae_item_std = float(np.sqrt(max(mae_item_var, 0.0)))
            mase_std = mae_item_std / naive_denom if naive_denom > 0.0 else float("nan")
        else:
            mase_std = float("nan")
    else:
        mase_mean = float("nan")
        mase_std = float("nan")
    crps_mean = float(np.mean(crps_per_sample)) if crps_per_sample else 0.0
    crps_std = float(np.std(crps_per_sample)) if crps_per_sample else 0.0

    per_horizon = {}
    for h, values in sorted(per_h_num.items(), key=lambda kv: kv[0]):
        den = values["den"]
        per_horizon[int(h)] = {
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
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "weighted_pinball_mean": pinball_mean,
            "weighted_pinball_std": pinball_std,
            "coverage": coverage,
            "interval_width": interval_width,
            "calibration_error": calibration_error,
            "wmape_mean": wmape_mean,
            "wmape_std": wmape_std,
            "mase_mean": mase_mean,
            "mase_std": mase_std,
            "crps_mean": crps_mean,
            "crps_std": crps_std,
        },
        "per_horizon": per_horizon,
        "per_region": per_region_out,
        "quantiles": quantiles,
        "metrics_original_scale": bool(args.metrics_original_scale),
        "metrics_style": str(args.metrics_style),
        "scaler_path": str(args.scaler_path) if args.scaler_path else None,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)

    print(f"Saved evaluation metrics to {output_path}")


if __name__ == "__main__":
    main()
