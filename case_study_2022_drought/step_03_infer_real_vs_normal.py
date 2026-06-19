#!/usr/bin/env python3
"""Step 03: inference for 2022 drought retrospective case-study.

Runs two states with same initial vegetation context:
- real_summer_2022: real future meteo
- simulated_summer_2022: DOY climatology replacement for temp/rain on JJA 2022

Outputs:
- predictions_real_summer_2022.csv
- predictions_simulated_summer_2022.csv
- metrics_real_summer_2022.json
- metrics_simulated_summer_2022.json
- metrics_mean_two_states.json
- delta_ndvi_summary_by_country.csv
- delta_ndvi_sample_level.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import lightning.pytorch as pl
except Exception:  # pragma: no cover
    import pytorch_lightning as pl

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.datamodule import CacheDataModule
from training.world_lightning_module import VegetationWorldLightningModule


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return True
    if s in {"0", "false", "no", "n", "f"}:
        return False
    raise ValueError(f"Cannot parse bool: {v}")


def parse_args():
    p = argparse.ArgumentParser(description="Step03 inference real_summer_2022 vs simulated_summer_2022")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache-root", action="append", required=True, help="Repeat for ood-t and ood-st")
    p.add_argument("--scaler-path", default="Data/cache/train_avg_NDVI_clear_sky/scaler.json")
    p.add_argument("--country-climatology-csv", required=True)
    p.add_argument("--global-climatology-csv", required=True)
    p.add_argument("--countries", nargs="+", default=["France", "Spain"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--apply-scaling", type=str2bool, default=True)
    p.add_argument("--feature-engineering", type=str2bool, default=True)
    p.add_argument("--min-crop-pixels", type=float, default=60.0)
    p.add_argument("--data-summary-path", type=str, default="Data/dataSummary_completed.csv")
    p.add_argument("--pin-memory", type=str2bool, default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="case_study_2022_drought/outputs/step03")
    p.add_argument(
        "--simulation-strategy",
        choices=["doy_climatology", "real_offset"],
        default="doy_climatology",
        help="How to build simulated summer forcing",
    )
    p.add_argument("--temp-offset-c", type=float, default=-4.0, help="Used when simulation-strategy=real_offset")
    p.add_argument("--rain-multiplier", type=float, default=1.4, help="Used when simulation-strategy=real_offset")
    p.add_argument(
        "--rain-additive",
        type=float,
        default=0.0,
        help="Additive rainfall term in raw units; used when simulation-strategy=real_offset",
    )
    return p.parse_args()


def _country_bbox(lat: float, lon: float) -> str:
    if 36.0 <= lat <= 44.8 and -9.8 <= lon <= 4.0:
        return "Spain"
    if 41.0 <= lat <= 51.6 and -5.6 <= lon <= 10.2:
        return "France"
    return "other"


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def _find_feature_idx(feature_names: list[str], name: str) -> int:
    low = [str(x).lower() for x in feature_names]
    return low.index(name.lower())


def _scale_feature_value(raw_value: float, scaler, feature_idx: int) -> float:
    if scaler is None:
        return float(raw_value)
    mode = str(getattr(scaler, "mode", ""))
    eps = float(getattr(scaler, "eps", 1e-5))
    if mode == "standardization":
        m = float(scaler.mean[feature_idx])
        v = float(scaler.var[feature_idx])
        return (float(raw_value) - m) / np.sqrt(v + eps)
    if mode == "standardization_arcsinh":
        m = float(scaler.mean[feature_idx])
        v = float(scaler.var[feature_idx])
        z = (float(raw_value) - m) / np.sqrt(v + eps)
        return float(np.arcsinh(z))
    if mode == "min-max":
        mn = float(scaler.min_value[feature_idx])
        mx = float(scaler.max_value[feature_idx])
        den = max(mx - mn, eps)
        return (float(raw_value) - mn) / den
    return float(raw_value)


def _inverse_scale_feature_value(scaled_value: float, scaler, feature_idx: int) -> float:
    if scaler is None:
        return float(scaled_value)
    mode = str(getattr(scaler, "mode", ""))
    eps = float(getattr(scaler, "eps", 1e-5))
    if mode == "standardization":
        m = float(scaler.mean[feature_idx])
        v = float(scaler.var[feature_idx])
        return float(scaled_value) * np.sqrt(v + eps) + m
    if mode == "standardization_arcsinh":
        m = float(scaler.mean[feature_idx])
        v = float(scaler.var[feature_idx])
        z = np.sinh(float(scaled_value))
        return z * np.sqrt(v + eps) + m
    if mode == "min-max":
        mn = float(scaler.min_value[feature_idx])
        mx = float(scaler.max_value[feature_idx])
        den = max(mx - mn, eps)
        return float(scaled_value) * den + mn
    return float(scaled_value)


def _inverse_scale_target_value(scaled_value: float, scaler) -> float:
    if scaler is None or not getattr(scaler, "has_target_stats", lambda: False)():
        return float(scaled_value)
    mode = str(getattr(scaler, "mode", ""))
    eps = float(getattr(scaler, "eps", 1e-5))
    if mode == "standardization":
        mean = float(np.asarray(scaler.target_mean).reshape(-1)[0])
        var = float(np.asarray(scaler.target_var).reshape(-1)[0])
        return float(scaled_value) * np.sqrt(var + eps) + mean
    if mode == "standardization_arcsinh":
        mean = float(np.asarray(scaler.target_mean).reshape(-1)[0])
        var = float(np.asarray(scaler.target_var).reshape(-1)[0])
        z = np.sinh(float(scaled_value))
        return z * np.sqrt(var + eps) + mean
    if mode == "min-max":
        mn = float(np.asarray(scaler.target_min_value).reshape(-1)[0])
        mx = float(np.asarray(scaler.target_max_value).reshape(-1)[0])
        den = max(mx - mn, eps)
        return float(scaled_value) * den + mn
    return float(scaled_value)


def _load_climatology(country_csv: Path, global_csv: Path):
    cdf = pd.read_csv(country_csv)
    gdf = pd.read_csv(global_csv)
    c_map: dict[tuple[str, int], tuple[float, float]] = {}
    for _, r in cdf.iterrows():
        c_map[(str(r["country"]), int(r["doy"]))] = (
            float(r["avg_temperature_median"]),
            float(r["rainfall_median"]),
        )
    g_map: dict[int, tuple[float, float]] = {}
    for _, r in gdf.iterrows():
        g_map[int(r["doy"])] = (float(r["avg_temperature_median"]), float(r["rainfall_median"]))
    return c_map, g_map


def _replace_with_simulated_summer(
    batch: dict[str, Any],
    *,
    temp_idx: int,
    rain_idx: int,
    scaler,
    c_map: dict[tuple[str, int], tuple[float, float]],
    g_map: dict[int, tuple[float, float]],
    allowed_countries: set[str],
) -> dict[str, Any]:
    out = dict(batch)
    fut = batch["future"].clone()
    bsz, seq_len, _ = fut.shape
    fut_pad = batch["future_pad_mask"]
    lats = batch["latitude"]
    lons = batch["longitude"]
    fut_ts = batch["future_timestamps"]

    for b in range(bsz):
        country = _country_bbox(float(lats[b]), float(lons[b]))
        if country not in allowed_countries:
            continue
        for t in range(seq_len):
            if bool(fut_pad[b, t].item()):
                continue
            ts = pd.to_datetime(str(fut_ts[b][t]), errors="coerce")
            if pd.isna(ts):
                continue
            if ts.year != 2022 or ts.month not in (6, 7, 8):
                continue
            doy = int(ts.dayofyear)
            pair = c_map.get((country, doy), g_map.get(doy, None))
            if pair is None:
                continue
            temp_raw, rain_raw = pair
            fut[b, t, temp_idx] = _scale_feature_value(temp_raw, scaler, temp_idx)
            fut[b, t, rain_idx] = _scale_feature_value(rain_raw, scaler, rain_idx)

    out["future"] = fut
    return out


def _replace_with_real_offset_summer(
    batch: dict[str, Any],
    *,
    temp_idx: int,
    rain_idx: int,
    scaler,
    allowed_countries: set[str],
    temp_offset_c: float,
    rain_multiplier: float,
    rain_additive: float,
) -> dict[str, Any]:
    out = dict(batch)
    fut = batch["future"].clone()
    bsz, seq_len, _ = fut.shape
    fut_pad = batch["future_pad_mask"]
    lats = batch["latitude"]
    lons = batch["longitude"]
    fut_ts = batch["future_timestamps"]

    for b in range(bsz):
        country = _country_bbox(float(lats[b]), float(lons[b]))
        if country not in allowed_countries:
            continue
        for t in range(seq_len):
            if bool(fut_pad[b, t].item()):
                continue
            ts = pd.to_datetime(str(fut_ts[b][t]), errors="coerce")
            if pd.isna(ts):
                continue
            if ts.year != 2022 or ts.month not in (6, 7, 8):
                continue

            # Convert scaled -> raw, apply offsets in raw physical units, convert back.
            temp_real_scaled = float(fut[b, t, temp_idx].item())
            rain_real_scaled = float(fut[b, t, rain_idx].item())
            temp_real_raw = _inverse_scale_feature_value(temp_real_scaled, scaler, temp_idx)
            rain_real_raw = _inverse_scale_feature_value(rain_real_scaled, scaler, rain_idx)

            temp_sim_raw = temp_real_raw + float(temp_offset_c)
            rain_sim_raw = max(0.0, rain_real_raw * float(rain_multiplier) + float(rain_additive))

            fut[b, t, temp_idx] = _scale_feature_value(temp_sim_raw, scaler, temp_idx)
            fut[b, t, rain_idx] = _scale_feature_value(rain_sim_raw, scaler, rain_idx)

    out["future"] = fut
    return out


def _quantile_pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    e = y - q
    return np.maximum(tau * e, (tau - 1.0) * e)


def _compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {k: float("nan") for k in ["mae", "rmse", "weighted_pinball", "wmape", "mase", "crps"]}

    y = df["gt"].to_numpy(dtype=np.float64)
    p10 = df["pred_q10"].to_numpy(dtype=np.float64)
    p50 = df["pred_q50"].to_numpy(dtype=np.float64)
    p90 = df["pred_q90"].to_numpy(dtype=np.float64)
    w = 1.0 / (1.0 + 0.1 * df["delta_days"].to_numpy(dtype=np.float64))

    abs_e = np.abs(p50 - y)
    sq_e = (p50 - y) ** 2
    mae = float(np.mean(abs_e))
    rmse = float(np.sqrt(np.mean(sq_e)))

    pin10 = _quantile_pinball(y, p10, 0.1)
    pin50 = _quantile_pinball(y, p50, 0.5)
    pin90 = _quantile_pinball(y, p90, 0.9)
    pin = np.stack([pin10, pin50, pin90], axis=-1).mean(axis=-1)
    weighted_pinball = float((pin * w).sum() / max(w.sum(), 1e-12))

    den_wmape = np.abs(y).sum()
    wmape = float(abs_e.sum() / den_wmape) if den_wmape > 1e-12 else float("nan")

    # Naive reference from consecutive observed targets within each sample.
    mase_num = []
    for _, g in df.sort_values(["sample_key", "target_ts"]).groupby("sample_key"):
        yy = g["gt"].to_numpy(dtype=np.float64)
        if len(yy) > 1:
            d = np.abs(np.diff(yy))
            if np.isfinite(d).any():
                mase_num.extend(d[np.isfinite(d)].tolist())
    naive = float(np.mean(mase_num)) if mase_num else float("nan")
    mase = float(mae / naive) if np.isfinite(naive) and naive > 1e-12 else float("nan")

    # 3-quantile CRPS approximation.
    taus = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    pin_mat = np.stack([pin10, pin50, pin90], axis=-1)
    crps = float(2.0 * np.trapz(pin_mat, taus, axis=-1).mean())

    return {
        "mae": mae,
        "rmse": rmse,
        "weighted_pinball": weighted_pinball,
        "wmape": wmape,
        "mase": mase,
        "crps": crps,
    }


def _merge_metric_dict(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    out = {}
    keys = sorted(set(a.keys()) | set(b.keys()))
    for k in keys:
        av = a.get(k, float("nan"))
        bv = b.get(k, float("nan"))
        if np.isfinite(av) and np.isfinite(bv):
            out[k] = float((av + bv) * 0.5)
        else:
            out[k] = float("nan")
    return out


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    c_map, g_map = _load_climatology(Path(args.country_climatology_csv), Path(args.global_climatology_csv))
    allowed_countries = {c.strip() for c in args.countries}

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = VegetationWorldLightningModule.load_from_checkpoint(str(ckpt), map_location=device)
    module.eval().to(device)
    q_levels = list(module.quantiles)
    q10_idx = q_levels.index(0.1) if 0.1 in q_levels else 0
    q50_idx = q_levels.index(0.5) if 0.5 in q_levels else len(q_levels) // 2
    q90_idx = q_levels.index(0.9) if 0.9 in q_levels else len(q_levels) - 1

    rows_real: list[dict[str, Any]] = []
    rows_sim: list[dict[str, Any]] = []

    for cache_root in args.cache_root:
        dm = CacheDataModule(
            cache_root=cache_root,
            scaler_path=args.scaler_path,
            predict_cache_root=cache_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            apply_scaling=args.apply_scaling,
            feature_engineering=args.feature_engineering,
            min_crop_pixels=args.min_crop_pixels,
            data_summary_path=args.data_summary_path,
            pin_memory=args.pin_memory,
        )
        dm.setup(stage="predict")
        loader = dm.predict_dataloader()
        if loader is None:
            continue

        feat_names = dm.feature_names
        if feat_names is None:
            raise RuntimeError(f"feature_names unavailable for {cache_root}")
        temp_idx = _find_feature_idx(feat_names, "avg_temperature")
        rain_idx = _find_feature_idx(feat_names, "rainfall")
        scaler = dm.predict_set.scaler if (dm.predict_set is not None and args.apply_scaling) else None

        with torch.no_grad():
            for batch in loader:
                # Keep metadata on CPU for rows.
                meta = {k: batch[k] for k in ["area", "source", "latitude", "longitude", "future_timestamps", "target_timestamps"]}
                batch_dev = _to_device(batch, device)

                out_real = module.model(batch_dev)
                if args.simulation_strategy == "doy_climatology":
                    batch_sim = _replace_with_simulated_summer(
                        batch=batch_dev,
                        temp_idx=temp_idx,
                        rain_idx=rain_idx,
                        scaler=scaler,
                        c_map=c_map,
                        g_map=g_map,
                        allowed_countries=allowed_countries,
                    )
                else:
                    batch_sim = _replace_with_real_offset_summer(
                        batch=batch_dev,
                        temp_idx=temp_idx,
                        rain_idx=rain_idx,
                        scaler=scaler,
                        allowed_countries=allowed_countries,
                        temp_offset_c=float(args.temp_offset_c),
                        rain_multiplier=float(args.rain_multiplier),
                        rain_additive=float(args.rain_additive),
                    )
                out_sim = module.model(batch_sim)

                q_real = out_real["quantiles"].detach().cpu().numpy()  # [B,L,C,Q]
                q_sim = out_sim["quantiles"].detach().cpu().numpy()
                fut_real = batch_dev["future"].detach().cpu().numpy()  # [B,L,F]
                fut_sim = batch_sim["future"].detach().cpu().numpy()   # [B,L,F]
                tgt = batch["target"].detach().cpu().numpy()  # [B,K] or [B,K,C]
                tgt_mask = batch["target_mask"].detach().cpu().numpy()  # bool True invalid
                pos = batch["future_target_positions"].detach().cpu().numpy()
                dd = batch["target_delta_days"].detach().cpu().numpy()

                if tgt.ndim == 2:
                    tgt = tgt[..., None]
                if tgt_mask.ndim == 2:
                    tgt_mask = tgt_mask[..., None]

                bsz, k_steps, c_dim = tgt.shape
                for b in range(bsz):
                    lat = float(meta["latitude"][b])
                    lon = float(meta["longitude"][b])
                    country = _country_bbox(lat, lon)
                    if country not in allowed_countries:
                        continue

                    sample_key = f"{cache_root}|{meta['area'][b]}|{meta['source'][b]}|{b}|{lat:.4f}|{lon:.4f}"
                    for j in range(k_steps):
                        p = int(pos[b, j])
                        ts = pd.to_datetime(str(meta["target_timestamps"][b][j]), errors="coerce")
                        if pd.isna(ts) or ts.year != 2022 or ts.month not in (6, 7, 8):
                            continue
                        for c in range(c_dim):
                            if bool(tgt_mask[b, j, c]):
                                continue
                            y = float(tgt[b, j, c])
                            r10 = float(q_real[b, p, c, q10_idx])
                            r50 = float(q_real[b, p, c, q50_idx])
                            r90 = float(q_real[b, p, c, q90_idx])
                            s10 = float(q_sim[b, p, c, q10_idx])
                            s50 = float(q_sim[b, p, c, q50_idx])
                            s90 = float(q_sim[b, p, c, q90_idx])
                            y_raw = _inverse_scale_target_value(y, scaler)
                            r10_raw = _inverse_scale_target_value(r10, scaler)
                            r50_raw = _inverse_scale_target_value(r50, scaler)
                            r90_raw = _inverse_scale_target_value(r90, scaler)
                            s10_raw = _inverse_scale_target_value(s10, scaler)
                            s50_raw = _inverse_scale_target_value(s50, scaler)
                            s90_raw = _inverse_scale_target_value(s90, scaler)
                            temp_real_scaled = float(fut_real[b, p, temp_idx])
                            temp_sim_scaled = float(fut_sim[b, p, temp_idx])
                            rain_real_scaled = float(fut_real[b, p, rain_idx])
                            rain_sim_scaled = float(fut_sim[b, p, rain_idx])
                            temp_real_raw = _inverse_scale_feature_value(temp_real_scaled, scaler, temp_idx)
                            temp_sim_raw = _inverse_scale_feature_value(temp_sim_scaled, scaler, temp_idx)
                            rain_real_raw = _inverse_scale_feature_value(rain_real_scaled, scaler, rain_idx)
                            rain_sim_raw = _inverse_scale_feature_value(rain_sim_scaled, scaler, rain_idx)
                            row_base = {
                                "cache_root": cache_root,
                                "country": country,
                                "area": str(meta["area"][b]),
                                "source": str(meta["source"][b]),
                                "sample_key": sample_key,
                                "lat": lat,
                                "lon": lon,
                                "target_ts": str(ts.date()),
                                "delta_days": float(dd[b, j]),
                                "channel": int(c),
                                "gt": y,
                                "gt_raw": y_raw,
                                "temp_real_scaled": temp_real_scaled,
                                "temp_simulated_scaled": temp_sim_scaled,
                                "temp_real_raw": temp_real_raw,
                                "temp_simulated_raw": temp_sim_raw,
                                "rain_real_scaled": rain_real_scaled,
                                "rain_simulated_scaled": rain_sim_scaled,
                                "rain_real_raw": rain_real_raw,
                                "rain_simulated_raw": rain_sim_raw,
                            }
                            rows_real.append(
                                {
                                    **row_base,
                                    "state": "real_summer_2022",
                                    "pred_q10": r10,
                                    "pred_q50": r50,
                                    "pred_q90": r90,
                                    "pred_q10_raw": r10_raw,
                                    "pred_q50_raw": r50_raw,
                                    "pred_q90_raw": r90_raw,
                                }
                            )
                            rows_sim.append(
                                {
                                    **row_base,
                                    "state": "simulated_summer_2022",
                                    "pred_q10": s10,
                                    "pred_q50": s50,
                                    "pred_q90": s90,
                                    "pred_q10_raw": s10_raw,
                                    "pred_q50_raw": s50_raw,
                                    "pred_q90_raw": s90_raw,
                                }
                            )

    df_real = pd.DataFrame(rows_real)
    df_sim = pd.DataFrame(rows_sim)
    if df_real.empty or df_sim.empty:
        raise RuntimeError("No valid JJA 2022 supervised rows found for requested countries.")

    real_csv = out_dir / "predictions_real_summer_2022.csv"
    sim_csv = out_dir / "predictions_simulated_summer_2022.csv"
    df_real.to_csv(real_csv, index=False)
    df_sim.to_csv(sim_csv, index=False)
    # Convenience exports in real NDVI space only.
    raw_cols = [
        "cache_root",
        "country",
        "area",
        "source",
        "sample_key",
        "lat",
        "lon",
        "target_ts",
        "delta_days",
        "channel",
        "gt_raw",
        "pred_q10_raw",
        "pred_q50_raw",
        "pred_q90_raw",
        "temp_real_raw",
        "temp_simulated_raw",
        "rain_real_raw",
        "rain_simulated_raw",
    ]
    df_real[raw_cols].rename(
        columns={"pred_q10_raw": "pred_q10", "pred_q50_raw": "pred_q50", "pred_q90_raw": "pred_q90", "gt_raw": "gt"}
    ).to_csv(out_dir / "predictions_real_summer_2022_raw.csv", index=False)
    df_sim[raw_cols].rename(
        columns={"pred_q10_raw": "pred_q10", "pred_q50_raw": "pred_q50", "pred_q90_raw": "pred_q90", "gt_raw": "gt"}
    ).to_csv(out_dir / "predictions_simulated_summer_2022_raw.csv", index=False)

    m_real = _compute_metrics(df_real)
    m_sim = _compute_metrics(df_sim)
    m_mean = _merge_metric_dict(m_real, m_sim)

    (out_dir / "metrics_real_summer_2022.json").write_text(json.dumps(m_real, indent=2))
    (out_dir / "metrics_simulated_summer_2022.json").write_text(json.dumps(m_sim, indent=2))
    (out_dir / "metrics_mean_two_states.json").write_text(json.dumps(m_mean, indent=2))
    (out_dir / "simulation_config.json").write_text(
        json.dumps(
            {
                "simulation_strategy": args.simulation_strategy,
                "temp_offset_c": float(args.temp_offset_c),
                "rain_multiplier": float(args.rain_multiplier),
                "rain_additive": float(args.rain_additive),
                "countries": list(args.countries),
            },
            indent=2,
        )
    )

    # Delta summary based on matched rows.
    key_cols = ["sample_key", "target_ts", "channel"]
    r = df_real[
        key_cols
        + [
            "country",
            "lat",
            "lon",
            "pred_q50",
            "gt",
            "temp_real_raw",
            "temp_simulated_raw",
            "rain_real_raw",
            "rain_simulated_raw",
        ]
    ].rename(columns={"pred_q50": "pred_q50_real"})
    s = df_sim[key_cols + ["pred_q50"]].rename(columns={"pred_q50": "pred_q50_simulated"})
    d = r.merge(s, on=key_cols, how="inner")
    d["delta_ndvi_q50"] = d["pred_q50_real"] - d["pred_q50_simulated"]
    d["error_real_abs"] = (d["pred_q50_real"] - d["gt"]).abs()
    d["error_simulated_abs"] = (d["pred_q50_simulated"] - d["gt"]).abs()
    d["delta_error_abs"] = d["error_real_abs"] - d["error_simulated_abs"]
    d.to_csv(out_dir / "delta_ndvi_sample_level.csv", index=False)

    summ = (
        d.groupby("country", as_index=False)
        .agg(
            n=("delta_ndvi_q50", "size"),
            delta_ndvi_q50_mean=("delta_ndvi_q50", "mean"),
            delta_ndvi_q50_median=("delta_ndvi_q50", "median"),
            delta_ndvi_q50_std=("delta_ndvi_q50", "std"),
            error_real_abs_mean=("error_real_abs", "mean"),
            error_simulated_abs_mean=("error_simulated_abs", "mean"),
            delta_error_abs_mean=("delta_error_abs", "mean"),
        )
        .sort_values("country")
    )
    summ.to_csv(out_dir / "delta_ndvi_summary_by_country.csv", index=False)

    np.savez_compressed(
        out_dir / "step03_delta_maps_input.npz",
        country=d["country"].to_numpy(dtype=object),
        latitude=d["lat"].to_numpy(dtype=np.float32),
        longitude=d["lon"].to_numpy(dtype=np.float32),
        delta_ndvi_q50=d["delta_ndvi_q50"].to_numpy(dtype=np.float32),
    )

    print(f"[OK] Saved {real_csv}")
    print(f"[OK] Saved {sim_csv}")
    print(f"[OK] Saved {out_dir / 'delta_ndvi_summary_by_country.csv'}")


if __name__ == "__main__":
    main()
