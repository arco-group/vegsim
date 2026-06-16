"""Scenario perturbation utilities for inference-time counterfactual simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class ScenarioChange:
    name: str
    precipitation_indices: list[int]
    temperature_indices: list[int]
    precipitation_multiplier: float | None
    temperature_additive: float | None


def _find_feature_indices(feature_names: list[str], keywords: tuple[str, ...]) -> list[int]:
    out: list[int] = []
    for idx, name in enumerate(feature_names):
        lower = str(name).lower()
        if any(key in lower for key in keywords):
            out.append(idx)
    return out


def _resolve_scenario_feature_indices(
    feature_names: list[str],
    scenario_config: dict,
) -> tuple[list[int], list[int]]:
    precip_idx = scenario_config.get("precipitation_feature_indices", None)
    temp_idx = scenario_config.get("temperature_feature_indices", None)

    if precip_idx is None:
        precip_idx = _find_feature_indices(
            feature_names,
            keywords=("rainfall", "precip", "rain", "tp"),
        )
    if temp_idx is None:
        temp_idx = _find_feature_indices(
            feature_names,
            keywords=("temperature", "temp", "t2m"),
        )

    if scenario_config.get("precipitation_feature_names"):
        names = {str(n).lower() for n in scenario_config["precipitation_feature_names"]}
        precip_idx = [i for i, n in enumerate(feature_names) if str(n).lower() in names]
    if scenario_config.get("temperature_feature_names"):
        names = {str(n).lower() for n in scenario_config["temperature_feature_names"]}
        temp_idx = [i for i, n in enumerate(feature_names) if str(n).lower() in names]

    return sorted(set(int(i) for i in precip_idx)), sorted(set(int(i) for i in temp_idx))


def perturb_future_covariates(
    x_future: torch.Tensor,
    feature_names: list[str],
    scenario_config: dict,
    *,
    future_mask: torch.Tensor | None = None,
    scenario_name: str = "scenario",
) -> tuple[torch.Tensor, ScenarioChange]:
    """Apply covariate perturbations to future drivers.

    Args:
        x_future: [B, L, C]
        feature_names: list of length C
        scenario_config: dict with optional keys:
            - precipitation_multiplier
            - temperature_additive
            - precipitation_feature_indices / names
            - temperature_feature_indices / names
        future_mask: [B, L, C] optional, True for invalid values (unchanged)
    """

    if x_future.dim() != 3:
        raise ValueError("x_future must be [B,L,C]")
    if len(feature_names) != x_future.shape[-1]:
        raise ValueError("feature_names length must match x_future feature dimension")

    precipitation_multiplier = scenario_config.get("precipitation_multiplier", None)
    temperature_additive = scenario_config.get("temperature_additive", None)

    precip_idx, temp_idx = _resolve_scenario_feature_indices(feature_names, scenario_config)

    out = x_future.clone()

    if precipitation_multiplier is not None and precip_idx:
        if future_mask is None:
            out[..., precip_idx] = out[..., precip_idx] * float(precipitation_multiplier)
        else:
            valid = ~future_mask[..., precip_idx]
            original = out[..., precip_idx]
            updated = original * float(precipitation_multiplier)
            out[..., precip_idx] = torch.where(valid, updated, original)

    if temperature_additive is not None and temp_idx:
        if future_mask is None:
            out[..., temp_idx] = out[..., temp_idx] + float(temperature_additive)
        else:
            valid = ~future_mask[..., temp_idx]
            original = out[..., temp_idx]
            updated = original + float(temperature_additive)
            out[..., temp_idx] = torch.where(valid, updated, original)

    change = ScenarioChange(
        name=scenario_name,
        precipitation_indices=precip_idx,
        temperature_indices=temp_idx,
        precipitation_multiplier=float(precipitation_multiplier) if precipitation_multiplier is not None else None,
        temperature_additive=float(temperature_additive) if temperature_additive is not None else None,
    )

    return out, change


_ENGINEERED_FEATURE_NAMES = [
    "rain_cum_between_targets",
    "temp_lt_10_between_targets",
    "temp_gt_30_between_targets",
    "rain_cum_7d",
    "rain_cum_14d",
    "temp_lt_10_7d",
    "temp_gt_30_7d",
    "temp_lt_10_14d",
    "temp_gt_30_14d",
]


def _between_targets_stats(sequence: np.ndarray, rain_idx: int | None, temp_idx: int | None, boundaries: list[int]):
    time_steps = sequence.shape[0]
    rain_feat = np.full((time_steps,), np.nan, dtype=np.float32)
    cold_feat = np.full((time_steps,), np.nan, dtype=np.float32)
    hot_feat = np.full((time_steps,), np.nan, dtype=np.float32)
    for prev, b in zip([-1] + boundaries[:-1], boundaries):
        if b >= time_steps or b < 0:
            continue
        window = sequence[prev + 1 : b + 1]
        rain_val = float(window[:, rain_idx].sum()) if rain_idx is not None else np.nan
        temp_col = window[:, temp_idx] if temp_idx is not None else np.full((window.shape[0],), np.nan, dtype=np.float32)
        cold_val = float((temp_col < 10.0).sum()) if temp_idx is not None else np.nan
        hot_val = float((temp_col > 30.0).sum()) if temp_idx is not None else np.nan
        rain_feat[b] = rain_val
        cold_feat[b] = cold_val
        hot_feat[b] = hot_val
    return rain_feat, cold_feat, hot_feat


def _rolling_stats(sequence: np.ndarray, timestamps: list[str], rain_idx: int | None, temp_idx: int | None, window_days: int):
    ts = np.array(timestamps, dtype="datetime64[ns]")
    rain_feat = np.zeros((len(ts),), dtype=np.float32)
    cold_feat = np.zeros((len(ts),), dtype=np.float32)
    hot_feat = np.zeros((len(ts),), dtype=np.float32)
    for i, t in enumerate(ts):
        start = t - np.timedelta64(window_days, "D")
        mask = (ts >= start) & (ts <= t)
        window = sequence[mask]
        if window.size == 0:
            continue
        if rain_idx is not None:
            rain_feat[i] = float(window[:, rain_idx].sum())
        if temp_idx is not None:
            temp_col = window[:, temp_idx]
            cold_feat[i] = float((temp_col < 10.0).sum())
            hot_feat[i] = float((temp_col > 30.0).sum())
    return rain_feat, cold_feat, hot_feat


def _recompute_engineered_future(
    history_seq: np.ndarray,
    history_mask: np.ndarray,
    history_timestamps: list[str],
    future_seq: np.ndarray,
    future_mask: np.ndarray,
    future_timestamps: list[str],
    future_target_positions: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    engineered_indices = [feature_names.index(name) for name in _ENGINEERED_FEATURE_NAMES if name in feature_names]
    if len(engineered_indices) != len(_ENGINEERED_FEATURE_NAMES):
        return future_seq, future_mask

    try:
        rain_idx = feature_names.index("rainfall")
    except ValueError:
        rain_idx = None
    try:
        temp_idx = feature_names.index("avg_temperature")
    except ValueError:
        temp_idx = None

    target_idx = len(feature_names) - 1
    history_bounds = [i for i in range(history_seq.shape[0]) if i < history_mask.shape[0] and not bool(history_mask[i, target_idx])]
    fut_pos = [int(x) for x in np.asarray(future_target_positions).reshape(-1).tolist()]
    fut_pos = sorted(set([x for x in fut_pos if 0 <= x < future_seq.shape[0]]))

    combined_seq = np.concatenate([history_seq, future_seq], axis=0)
    combined_ts = list(history_timestamps) + list(future_timestamps)
    combined_bounds = history_bounds + [history_seq.shape[0] + b for b in fut_pos]

    rain_bt, cold_bt, hot_bt = _between_targets_stats(combined_seq, rain_idx, temp_idx, combined_bounds)
    rain7, cold7, hot7 = _rolling_stats(combined_seq, combined_ts, rain_idx, temp_idx, 7)
    rain14, cold14, hot14 = _rolling_stats(combined_seq, combined_ts, rain_idx, temp_idx, 14)
    engineered = np.stack([rain_bt, cold_bt, hot_bt, rain7, rain14, cold7, hot7, cold14, hot14], axis=1).astype(np.float32)
    eng_f = engineered[history_seq.shape[0] :]

    out_f = future_seq.copy()
    out_m = future_mask.copy()
    for j, feat_idx in enumerate(engineered_indices):
        out_f[:, feat_idx] = eng_f[:, j]
        out_m[:, feat_idx] = np.isnan(eng_f[:, j])
    return out_f, out_m


def _inverse_with_mask(array: np.ndarray, mask: np.ndarray | None, scaler: Any | None) -> np.ndarray:
    if scaler is None:
        return array
    restored = scaler.inverse_transform(array).astype(np.float32)
    if mask is not None:
        restored[mask] = array[mask]
    return restored


def _forward_with_mask(array: np.ndarray, mask: np.ndarray | None, scaler: Any | None) -> np.ndarray:
    if scaler is None:
        return array
    transformed = scaler.transform(array).astype(np.float32)
    if mask is not None:
        transformed[mask] = array[mask]
    return transformed


def perturb_future_covariates_physical(
    batch: dict[str, Any],
    feature_names: list[str],
    scenario_config: dict,
    *,
    scaler: Any | None = None,
    scenario_name: str = "scenario",
) -> tuple[torch.Tensor, ScenarioChange]:
    """Apply scenario in raw physical space, then return model-space future tensor.

    Steps:
    1) Inverse transform history/future if scaler is provided.
    2) Perturb raw future covariates.
    3) Recompute engineered weather-derived features if present.
    4) Transform back to model space.
    """

    device = batch["future"].device
    dtype = batch["future"].dtype

    hist = batch["history"].detach().cpu().numpy().astype(np.float32)
    hist_mask = batch["history_mask"].detach().cpu().numpy().astype(bool)
    fut = batch["future"].detach().cpu().numpy().astype(np.float32)
    fut_mask = batch["future_mask"].detach().cpu().numpy().astype(bool)
    fut_pos = batch["future_target_positions"].detach().cpu().numpy()

    hist_raw = np.empty_like(hist)
    fut_raw = np.empty_like(fut)
    for i in range(hist.shape[0]):
        hist_raw[i] = _inverse_with_mask(hist[i], hist_mask[i], scaler)
        fut_raw[i] = _inverse_with_mask(fut[i], fut_mask[i], scaler)

    fut_raw_t = torch.from_numpy(fut_raw)
    fut_mask_t = torch.from_numpy(fut_mask)
    x_pert_raw_t, change = perturb_future_covariates(
        x_future=fut_raw_t,
        feature_names=feature_names,
        scenario_config=scenario_config,
        future_mask=fut_mask_t,
        scenario_name=scenario_name,
    )
    x_pert_raw = x_pert_raw_t.numpy().astype(np.float32)

    hist_ts_batch = batch.get("history_timestamps", None)
    fut_ts_batch = batch.get("future_timestamps", None)
    if hist_ts_batch is not None and fut_ts_batch is not None:
        # Start from perturbed tensors and overwrite only non-padded valid windows.
        x_recomputed = x_pert_raw.copy()
        mask_recomputed = fut_mask.copy()
        for i in range(x_pert_raw.shape[0]):
            hist_ts_i = list(hist_ts_batch[i])
            fut_ts_i = list(fut_ts_batch[i])
            hist_len = len(hist_ts_i)
            fut_len = len(fut_ts_i)
            if hist_len <= 0 or fut_len <= 0:
                continue

            x_i, m_i = _recompute_engineered_future(
                history_seq=hist_raw[i, :hist_len],
                history_mask=hist_mask[i, :hist_len],
                history_timestamps=hist_ts_i,
                future_seq=x_pert_raw[i, :fut_len],
                future_mask=fut_mask[i, :fut_len],
                future_timestamps=fut_ts_i,
                future_target_positions=fut_pos[i],
                feature_names=feature_names,
            )
            x_recomputed[i, :fut_len] = x_i
            mask_recomputed[i, :fut_len] = m_i
        x_pert_raw = x_recomputed
        fut_mask = mask_recomputed

    x_model = np.empty_like(x_pert_raw)
    for i in range(x_pert_raw.shape[0]):
        x_model[i] = _forward_with_mask(x_pert_raw[i], fut_mask[i], scaler)

    x_out = torch.from_numpy(x_model).to(device=device, dtype=dtype)
    return x_out, change
