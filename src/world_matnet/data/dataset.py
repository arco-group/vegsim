"""PyTorch dataset for cached time-series samples."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .metadata import extract_area_code
from .scaler import Scaler


class CacheTimeSeriesDataset(Dataset):
    def __init__(
        self,
        cache_dir,
        apply_scaling=False,
        scaler_path=None,
        data_summary_path="Data/dataSummary_completed.csv",
        min_crop_pixels=0.0,
        feature_engineering=False,
        discretize_target=False,
    ):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")

        self.apply_scaling = apply_scaling
        self.feature_engineering = feature_engineering
        self.discretize_target = discretize_target
        self.scaler_path = scaler_path or (self.cache_dir / "scaler.json")
        self.scaler = None
        self.min_crop_pixels = float(min_crop_pixels)
        self.crop_pixels_map = (
            self.load_crop_pixels_map(data_summary_path) if self.min_crop_pixels > 0 else {}
        )

        (
            self.history_sequences,
            self.history_masks,
            self.history_timestamps,
            self.future_sequences,
            self.future_masks,
            self.future_noise,
            self.future_timestamps,
            self.future_target_positions,
            self.targets,
            self.target_masks,
            self.target_timestamps,
            self.climates,
            self.latitudes,
            self.longitudes,
            self.crop_types,
            self.feature_names,
            self.metadata_records,
        ) = self.load_cache_files()

        self.engineered_stats = None
        if self.feature_engineering:
            (
                self.history_sequences,
                self.history_masks,
                self.future_sequences,
                self.future_masks,
                self.feature_names,
                self.engineered_stats,
            ) = self.add_engineered_features(
                self.history_sequences,
                self.history_masks,
                self.future_sequences,
                self.future_masks,
                self.history_timestamps,
                self.future_timestamps,
                self.future_target_positions,
                self.feature_names,
            )

        self.target_classes = None
        self.history_target_classes = None
        self.history_target_class_masks = None
        if self.discretize_target:
            (
                self.target_classes,
                self.history_target_classes,
                self.history_target_class_masks,
            ) = self.discretize_targets(
                self.targets,
                self.target_masks,
                self.history_sequences,
                self.history_masks,
                self.future_sequences,
                self.future_masks,
                self.feature_names,
            )

        if self.apply_scaling:
            self.load_scaler()
            if self.feature_engineering and self.engineered_stats is not None:
                self.extend_scaler_with_engineered(self.engineered_stats)
            self.history_sequences = self.transform_sequences(self.history_sequences)
            self.future_sequences = self.transform_sequences(self.future_sequences)
            self.targets = self.transform_targets(self.targets, self.target_masks)

    def load_cache_files(self):
        history_sequences = []
        history_masks = []
        history_timestamps = []
        future_sequences = []
        future_masks = []
        future_noise = []
        future_timestamps = []
        future_target_positions = []
        targets = []
        target_masks = []
        target_timestamps = []
        climates = []
        latitudes = []
        longitudes = []
        crop_types = []
        feature_names = None
        metadata_records = []

        cache_files = sorted(self.cache_dir.glob("*.npz"))
        if not cache_files:
            raise RuntimeError(f"No .npz cache files found in {self.cache_dir}")

        for cache_file in cache_files:
            data = np.load(cache_file, allow_pickle=True)
            metadata_path = cache_file.with_suffix(".json")
            with metadata_path.open("r", encoding="utf-8") as fp:
                metadata = json.load(fp)

            records = metadata.get("metadata_records", [])
            if not records:
                continue

            area_name = records[0]["area"]
            if self.min_crop_pixels > 0:
                area_crop = self.crop_pixels_map.get(area_name)
                if area_crop is None:
                    continue
                if area_crop < self.min_crop_pixels:
                    continue

            history_sequences.extend([np.asarray(seq, dtype=np.float32) for seq in data["history"].tolist()])
            history_masks.extend([np.asarray(mask, dtype=bool) for mask in data["history_mask"].tolist()])
            history_timestamps.extend(data["history_timestamps"].tolist())

            future_sequences.extend([np.asarray(seq, dtype=np.float32) for seq in data["future"].tolist()])
            future_masks.extend([np.asarray(mask, dtype=bool) for mask in data["future_mask"].tolist()])
            future_noise.extend([np.asarray(noise, dtype=np.float32) for noise in data["future_noise"].tolist()])
            future_timestamps.extend(data["future_timestamps"].tolist())

            if "future_target_positions" in data:
                future_target_positions.extend(data["future_target_positions"].tolist())
            else:
                fallback = []
                target_len = data["target"].shape[1]
                for _ in data["future"]:
                    fallback.append(list(range(target_len)))
                future_target_positions.extend(fallback)

            targets.append(data["target"].astype(np.float32))
            target_masks.append(data["target_mask"].astype(bool))
            target_timestamps.extend(data["target_timestamps"].tolist())

            climates.extend(self._read_string_array(data, "climate", records, "climate", default="unknown"))
            latitudes.extend(self._read_numeric_array(data, "latitude", records, "latitude"))
            longitudes.extend(self._read_numeric_array(data, "longitude", records, "longitude"))
            crop_types.extend(self._read_string_array(data, "crop_type", records, "crop_type", default="unknown"))

            feature_names = metadata["feature_names"]
            metadata_records.extend(records)

        targets_array = np.concatenate(targets, axis=0)
        target_masks_array = np.concatenate(target_masks, axis=0)

        return (
            history_sequences,
            history_masks,
            history_timestamps,
            future_sequences,
            future_masks,
            future_noise,
            future_timestamps,
            future_target_positions,
            targets_array,
            target_masks_array,
            target_timestamps,
            climates,
            latitudes,
            longitudes,
            crop_types,
            feature_names,
            metadata_records,
        )

    @staticmethod
    def _read_string_array(data, key, records, record_key, default="unknown"):
        if key in data:
            values = data[key].tolist()
            return [default if str(v).strip() == "" else str(v) for v in values]
        return [str(rec.get(record_key, default)) for rec in records]

    @staticmethod
    def _read_numeric_array(data, key, records, record_key):
        if key in data:
            values = np.asarray(data[key], dtype=np.float32)
            return values.tolist()
        out = []
        for rec in records:
            value = rec.get(record_key, None)
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                out.append(float("nan"))
        return out

    @staticmethod
    def _get_feature_indices(feature_names):
        def find_idx(name):
            try:
                return feature_names.index(name)
            except ValueError:
                return None

        rain_idx = find_idx("rainfall")
        temp_idx = find_idx("avg_temperature")
        target_idx = len(feature_names) - 1 if feature_names else None
        return rain_idx, temp_idx, target_idx

    @staticmethod
    def _discretize_array(values, mask):
        classes = np.full(values.shape, -1, dtype=np.int64)
        valid = ~mask
        if valid.any():
            v = values
            classes[valid & (v < 0.0)] = 0
            classes[valid & (v >= 0.0) & (v < 0.33)] = 1
            classes[valid & (v >= 0.33) & (v < 0.66)] = 2
            classes[valid & (v >= 0.66)] = 3
        return classes

    @staticmethod
    def _between_targets_stats(sequence, rain_idx, temp_idx, boundaries):
        time_steps = sequence.shape[0]
        rain_feat = np.full((time_steps,), np.nan, dtype=np.float32)
        cold_feat = np.full((time_steps,), np.nan, dtype=np.float32)
        hot_feat = np.full((time_steps,), np.nan, dtype=np.float32)
        for prev, b in zip([-1] + boundaries[:-1], boundaries):
            if b >= time_steps:
                continue
            window = sequence[prev + 1 : b + 1]
            rain_val = float(window[:, rain_idx].sum()) if rain_idx is not None else np.nan
            temp_col = window[:, temp_idx] if temp_idx is not None else np.full_like(window[:, 0], np.nan)
            cold_val = float((temp_col < 10.0).sum()) if temp_idx is not None else np.nan
            hot_val = float((temp_col > 30.0).sum()) if temp_idx is not None else np.nan
            rain_feat[b] = rain_val
            cold_feat[b] = cold_val
            hot_feat[b] = hot_val
        return rain_feat, cold_feat, hot_feat

    @staticmethod
    def _rolling_stats(sequence, timestamps, rain_idx, temp_idx, window_days):
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

    def add_engineered_features(
        self,
        history_sequences,
        history_masks,
        future_sequences,
        future_masks,
        history_timestamps,
        future_timestamps,
        future_target_positions,
        feature_names,
    ):
        rain_idx, temp_idx, target_idx = self._get_feature_indices(feature_names)

        new_feature_names = [
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

        def build_engineered(seq, ts, bounds):
            rain_bt, cold_bt, hot_bt = self._between_targets_stats(seq, rain_idx, temp_idx, bounds)
            rain7, cold7, hot7 = self._rolling_stats(seq, ts, rain_idx, temp_idx, 7)
            rain14, cold14, hot14 = self._rolling_stats(seq, ts, rain_idx, temp_idx, 14)
            engineered = np.stack(
                [rain_bt, cold_bt, hot_bt, rain7, rain14, cold7, hot7, cold14, hot14], axis=1
            ).astype(np.float32)
            return engineered

        history_bounds = []
        for seq, mask in zip(history_sequences, history_masks):
            valid_idx = [i for i in range(len(seq)) if i < mask.shape[0] and not mask[i, target_idx]]
            history_bounds.append(valid_idx)
        future_bounds = future_target_positions

        updated_history_seq = []
        updated_history_mask = []
        updated_future_seq = []
        updated_future_mask = []

        stats_sum = np.zeros(len(new_feature_names), dtype=np.float64)
        stats_sumsq = np.zeros(len(new_feature_names), dtype=np.float64)
        stats_min = np.full(len(new_feature_names), np.inf, dtype=np.float64)
        stats_max = np.full(len(new_feature_names), -np.inf, dtype=np.float64)
        stats_count = np.zeros(len(new_feature_names), dtype=np.int64)

        total = len(history_sequences)
        iterator = zip(
            history_sequences,
            history_masks,
            history_timestamps,
            history_bounds,
            future_sequences,
            future_masks,
            future_timestamps,
            future_bounds,
        )
        for seq_h, mask_h, ts_h, bounds_h, seq_f, mask_f, ts_f, bounds_f in tqdm(
            iterator,
            total=total,
            desc="Feature engineering",
            leave=False,
        ):
            eng_h = build_engineered(seq_h, ts_h, bounds_h)
            new_seq_h = np.concatenate(
                [seq_h[:, :target_idx], eng_h, seq_h[:, target_idx : target_idx + 1]],
                axis=1,
            )
            new_mask_cols_h = np.isnan(eng_h)
            new_mask_h = np.concatenate(
                [mask_h[:, :target_idx], new_mask_cols_h, mask_h[:, target_idx : target_idx + 1]],
                axis=1,
            )
            updated_history_seq.append(new_seq_h)
            updated_history_mask.append(new_mask_h)

            combined_seq = np.concatenate([seq_h, seq_f], axis=0)
            combined_ts = np.concatenate(
                [np.array(ts_h, dtype="datetime64[ns]"), np.array(ts_f, dtype="datetime64[ns]")]
            )
            combined_bounds = bounds_h + [len(seq_h) + b for b in bounds_f]
            eng_combined = build_engineered(combined_seq, combined_ts, combined_bounds)
            eng_f = eng_combined[len(seq_h) :]

            new_seq_f = np.concatenate(
                [seq_f[:, :target_idx], eng_f, seq_f[:, target_idx : target_idx + 1]],
                axis=1,
            )
            new_mask_cols_f = np.isnan(eng_f)
            new_mask_f = np.concatenate(
                [mask_f[:, :target_idx], new_mask_cols_f, mask_f[:, target_idx : target_idx + 1]],
                axis=1,
            )
            updated_future_seq.append(new_seq_f)
            updated_future_mask.append(new_mask_f)

            for j in range(len(new_feature_names)):
                for eng_block in (eng_h, eng_f):
                    col = eng_block[:, j]
                    valid = ~np.isnan(col)
                    if not valid.any():
                        continue
                    values = col[valid].astype(np.float64)
                    stats_sum[j] += values.sum()
                    stats_sumsq[j] += (values**2).sum()
                    stats_min[j] = min(stats_min[j], float(values.min()))
                    stats_max[j] = max(stats_max[j], float(values.max()))
                    stats_count[j] += len(values)

        base_names = feature_names[:target_idx]
        target_name = feature_names[target_idx]
        updated_feature_names = base_names + new_feature_names + [target_name]

        engineered_stats = None
        valid_counts = stats_count > 0
        if valid_counts.any():
            mean = np.full(len(new_feature_names), np.nan, dtype=np.float32)
            var = np.full(len(new_feature_names), np.nan, dtype=np.float32)
            min_v = np.full(len(new_feature_names), np.nan, dtype=np.float32)
            max_v = np.full(len(new_feature_names), np.nan, dtype=np.float32)
            mean[valid_counts] = (stats_sum[valid_counts] / stats_count[valid_counts]).astype(np.float32)
            var[valid_counts] = (
                stats_sumsq[valid_counts] / stats_count[valid_counts] - mean[valid_counts] ** 2
            ).astype(np.float32)
            min_v[valid_counts] = stats_min[valid_counts].astype(np.float32)
            max_v[valid_counts] = stats_max[valid_counts].astype(np.float32)
            engineered_stats = {"mean": mean, "var": var, "min": min_v, "max": max_v}

        return (
            updated_history_seq,
            updated_history_mask,
            updated_future_seq,
            updated_future_mask,
            updated_feature_names,
            engineered_stats,
        )

    def discretize_targets(
        self,
        targets,
        target_masks,
        history_sequences,
        history_masks,
        future_sequences,
        future_masks,
        feature_names,
    ):
        _, _, target_idx = self._get_feature_indices(feature_names)
        if target_idx is None:
            return None, None, None

        target_classes = self._discretize_array(targets, target_masks)

        history_classes = []
        history_class_masks = []
        for seq, mask in zip(history_sequences, history_masks):
            vals = seq[:, target_idx]
            mask_col = mask[:, target_idx].copy()
            history_classes.append(self._discretize_array(vals, mask_col))
            history_class_masks.append(mask_col)
            seq[:, target_idx] = 0.0
            mask[:, target_idx] = True

        for seq, mask in zip(future_sequences, future_masks):
            if target_idx < seq.shape[1]:
                seq[:, target_idx] = 0.0
            if target_idx < mask.shape[1]:
                mask[:, target_idx] = True

        return target_classes, history_classes, history_class_masks

    def load_scaler(self):
        if not Path(self.scaler_path).exists():
            raise FileNotFoundError(f"Scaler requested but not found in {self.scaler_path}")
        self.scaler = Scaler.load(self.scaler_path)

    def extend_scaler_with_engineered(self, engineered_stats):
        if self.scaler is None or engineered_stats is None:
            return
        insert_len = len(engineered_stats["mean"])
        if insert_len == 0:
            return

        if self.scaler.mode in {"standardization", "standardization_arcsinh"}:
            base_len = len(self.scaler.mean) - 1
            self.scaler.mean = np.concatenate(
                [self.scaler.mean[:base_len], engineered_stats["mean"], self.scaler.mean[base_len:]]
            )
            self.scaler.var = np.concatenate(
                [self.scaler.var[:base_len], engineered_stats["var"], self.scaler.var[base_len:]]
            )
        elif self.scaler.mode == "min-max":
            base_len = len(self.scaler.min_value) - 1
            self.scaler.min_value = np.concatenate(
                [self.scaler.min_value[:base_len], engineered_stats["min"], self.scaler.min_value[base_len:]]
            )
            self.scaler.max_value = np.concatenate(
                [self.scaler.max_value[:base_len], engineered_stats["max"], self.scaler.max_value[base_len:]]
            )
        else:
            raise ValueError("Unsupported scaler mode")

    def transform_sequences(self, sequences):
        if not self.scaler:
            return sequences

        if self.scaler.mode in {"standardization", "standardization_arcsinh"}:
            stats_len = len(self.scaler.mean)
        else:
            stats_len = len(self.scaler.min_value)

        transformed = []
        for seq in sequences:
            if seq.shape[1] != stats_len:
                raise ValueError(
                    f"Scaler defined for {stats_len} features, but sequence has {seq.shape[1]} features"
                )
            transformed.append(self.scaler.transform(seq).astype(np.float32))
        return transformed

    def transform_targets(self, targets, target_masks):
        if not self.scaler or not self.scaler.has_target_stats():
            return targets
        return self.scaler.transform_target(targets, target_masks)

    def __len__(self):
        return len(self.history_sequences)

    def __getitem__(self, index):
        meta = self.metadata_records[index]
        out = {
            "history": torch.tensor(self.history_sequences[index], dtype=torch.float32),
            "history_mask": torch.tensor(self.history_masks[index], dtype=torch.bool),
            "history_timestamps": self.history_timestamps[index],
            "future": torch.tensor(self.future_sequences[index], dtype=torch.float32),
            "future_mask": torch.tensor(self.future_masks[index], dtype=torch.bool),
            "future_noise": torch.tensor(self.future_noise[index], dtype=torch.float32),
            "future_timestamps": self.future_timestamps[index],
            "future_target_positions": torch.tensor(self.future_target_positions[index], dtype=torch.long),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
            "target_mask": torch.tensor(self.target_masks[index], dtype=torch.bool),
            "target_timestamps": self.target_timestamps[index],
            "climate": self.climates[index],
            "latitude": float(self.latitudes[index]) if np.isfinite(self.latitudes[index]) else float("nan"),
            "longitude": float(self.longitudes[index]) if np.isfinite(self.longitudes[index]) else float("nan"),
            "crop_type": self.crop_types[index],
            "area": meta["area"],
            "source": meta["csv"],
            "history_start": meta["history_start"],
            "history_end": meta["history_end"],
            "future_start": meta["future_start"],
            "future_end": meta["future_end"],
        }
        if self.discretize_target:
            if self.target_classes is None or self.history_target_classes is None:
                raise ValueError("discretize_target enabled but classes were not computed")
            out["target_class"] = torch.tensor(self.target_classes[index], dtype=torch.long)
            out["target_class_mask"] = torch.tensor(self.target_masks[index], dtype=torch.bool)
            out["history_target_class"] = torch.tensor(self.history_target_classes[index], dtype=torch.long)
            out["history_target_class_mask"] = torch.tensor(self.history_target_class_masks[index], dtype=torch.bool)
        return out

    def load_crop_pixels_map(self, summary_path):
        path = Path(summary_path)
        if not path.exists():
            return {}

        lookup = {}
        with path.open("r", encoding="utf-8") as fp:
            reader = csv.reader(fp)
            try:
                header = next(reader)
            except StopIteration:
                return lookup
            header = [h.strip().strip('"') for h in header]
            if "path_to_catalogue" not in header or "crop_pixels" not in header:
                return lookup
            idx_path = header.index("path_to_catalogue")
            idx_crop = header.index("crop_pixels")

            for row in reader:
                if not row:
                    continue
                padded = row + [""] * max(0, len(header) - len(row))
                path_value = padded[idx_path].strip().strip('"')
                crop_val = padded[idx_crop].strip().strip('"')
                if not path_value or not crop_val:
                    continue
                try:
                    crop_float = float(crop_val)
                except ValueError:
                    continue

                area_name = extract_area_code(path_value) or extract_area_code(Path(path_value).name)
                if area_name is None:
                    basename = Path(path_value.replace("\\", "/")).name
                    parts = basename.split("_")
                    if len(parts) >= 3:
                        area_name = parts[2]
                if not area_name:
                    continue

                stored_val = lookup.get(area_name)
                if stored_val is None or crop_float > stored_val:
                    lookup[area_name] = crop_float
        return lookup
