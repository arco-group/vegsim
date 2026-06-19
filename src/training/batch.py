"""Batch collation helpers for variable-length vegetation sequences."""

import hashlib

import numpy as np
import torch


def _stable_hash_to_int(value: str) -> int:
    digest = hashlib.md5(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def _ensure_target_3d(target: torch.Tensor, target_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if target.dim() == 2:
        target = target.unsqueeze(-1)
    if target_mask.dim() == 2:
        target_mask = target_mask.unsqueeze(-1)
    if target.dim() != 3 or target_mask.dim() != 3:
        raise ValueError("target and target_mask must be [B,K] or [B,K,C]")
    return target, target_mask


def _scatter_sparse_targets_to_dense(
    target: torch.Tensor,
    target_mask: torch.Tensor,
    future_target_positions: torch.Tensor,
    future_pad_mask: torch.Tensor,
    max_future: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter sparse supervised targets [B,K,C] to dense rollout axis [B,L,C]."""
    target, target_mask = _ensure_target_3d(target, target_mask)
    bsz, k_steps, c_target = target.shape

    dense_target = torch.zeros((bsz, max_future, c_target), dtype=target.dtype, device=target.device)
    dense_mask = torch.ones((bsz, max_future, c_target), dtype=torch.bool, device=target.device)

    for b in range(bsz):
        for j in range(k_steps):
            pos = int(future_target_positions[b, j].item())
            if pos < 0 or pos >= max_future:
                continue
            dense_target[b, pos] = target[b, j]
            dense_mask[b, pos] = target_mask[b, j]

    dense_mask = dense_mask | future_pad_mask.unsqueeze(-1)
    return dense_target, dense_mask


def collate_variable(batch):
    list_keys = {
        "history_timestamps",
        "future_timestamps",
        "target_timestamps",
        "climate",
        "latitude",
        "longitude",
        "crop_type",
        "area",
        "source",
        "history_start",
        "history_end",
        "future_start",
        "future_end",
    }

    collated = {}
    for key in list_keys:
        collated[key] = [item[key] for item in batch]

    history_lengths = [item["history"].shape[0] for item in batch]
    future_lengths = [item["future"].shape[0] for item in batch]
    max_history = max(history_lengths)
    max_future = max(future_lengths)

    padded_history = []
    padded_history_mask = []
    padded_future = []
    padded_future_mask = []
    padded_future_noise = []
    target_delta_days = []
    future_delta_days = []

    for item in batch:
        history = item["history"]
        history_mask = item["history_mask"]
        future = item["future"]
        future_mask = item["future_mask"]
        future_noise = item["future_noise"]

        history_end = np.datetime64(item["history_timestamps"][-1])
        target_ts = np.array(item["target_timestamps"], dtype="datetime64[ns]")
        delta_days = (target_ts - history_end).astype("timedelta64[D]").astype(np.float32)
        target_delta_days.append(torch.tensor(delta_days, dtype=torch.float32))
        future_ts = np.array(item["future_timestamps"], dtype="datetime64[ns]")
        delta_days_future = (future_ts - history_end).astype("timedelta64[D]").astype(np.float32)

        pad_rows_history = max_history - history.shape[0]
        if pad_rows_history > 0:
            history = torch.nn.functional.pad(history, (0, 0, 0, pad_rows_history))
            history_mask = torch.nn.functional.pad(history_mask, (0, 0, 0, pad_rows_history), value=True)
        padded_history.append(history)
        padded_history_mask.append(history_mask)

        pad_rows_future = max_future - future.shape[0]
        if pad_rows_future > 0:
            future = torch.nn.functional.pad(future, (0, 0, 0, pad_rows_future))
            future_mask = torch.nn.functional.pad(future_mask, (0, 0, 0, pad_rows_future), value=True)
            future_noise = torch.nn.functional.pad(future_noise, (0, 0, 0, pad_rows_future))
            delta_days_future = np.pad(delta_days_future, (0, pad_rows_future), mode="constant", constant_values=0.0)
        padded_future.append(future)
        padded_future_mask.append(future_mask)
        padded_future_noise.append(future_noise)
        future_delta_days.append(torch.tensor(delta_days_future, dtype=torch.float32))

    collated["history"] = torch.stack(padded_history)
    collated["history_mask"] = torch.stack(padded_history_mask)
    collated["history_pad_mask"] = torch.tensor(
        [[i >= length for i in range(max_history)] for length in history_lengths],
        dtype=torch.bool,
    )

    collated["future"] = torch.stack(padded_future)
    collated["future_mask"] = torch.stack(padded_future_mask)
    collated["future_noise"] = torch.stack(padded_future_noise)
    collated["future_pad_mask"] = torch.tensor(
        [[i >= length for i in range(max_future)] for length in future_lengths],
        dtype=torch.bool,
    )

    collated["future_target_positions"] = torch.stack([item["future_target_positions"] for item in batch])
    collated["target"] = torch.stack([item["target"] for item in batch])
    collated["target_mask"] = torch.stack([item["target_mask"] for item in batch])
    collated["target_delta_days"] = torch.stack(target_delta_days)
    collated["future_delta_days"] = torch.stack(future_delta_days)

    # Derived views used by the world-model family.
    history = collated["history"]
    future = collated["future"]
    history_mask = collated["history_mask"]
    future_mask = collated["future_mask"]

    if history.shape[-1] >= 1:
        collated["x_hist"] = history[..., :-1]
        collated["y_hist"] = history[..., -1:].contiguous()
        collated["x_future"] = future[..., :-1]
    else:
        collated["x_hist"] = history
        collated["y_hist"] = history
        collated["x_future"] = future

    collated["hist_mask"] = collated["history_pad_mask"] | history_mask.all(dim=-1)
    collated["horizons"] = collated["future_delta_days"]

    dense_target, dense_mask = _scatter_sparse_targets_to_dense(
        target=collated["target"],
        target_mask=collated["target_mask"],
        future_target_positions=collated["future_target_positions"],
        future_pad_mask=collated["future_pad_mask"],
        max_future=max_future,
    )
    collated["target_dense"] = dense_target
    collated["target_dense_mask"] = dense_mask

    spatial_cont = torch.tensor(
        [[float(lat), float(lon)] for lat, lon in zip(collated["latitude"], collated["longitude"])],
        dtype=torch.float32,
    )
    collated["spatial_cont"] = spatial_cont

    climate_ids = torch.tensor([_stable_hash_to_int(v) for v in collated["climate"]], dtype=torch.long)
    crop_ids = torch.tensor([_stable_hash_to_int(v) for v in collated["crop_type"]], dtype=torch.long)
    collated["spatial_cat"] = torch.stack([climate_ids, crop_ids], dim=-1)

    return collated


def masked_time_weighted_mse(preds, targets, mask, delta_days, alpha):
    weights = 1.0 / (1.0 + alpha * delta_days)
    weights = weights * (~mask).float()
    denom = weights.sum().clamp(min=1e-8)
    return ((preds - targets) ** 2 * weights).sum() / denom
