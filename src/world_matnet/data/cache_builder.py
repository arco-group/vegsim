"""Monthly-noise cache builder for time-series datasets."""

from __future__ import annotations

import argparse
import json
import logging
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from .constants import VEGETATION_COLUMNS
from .metadata import SampleMetadata, load_metadata_lookup
from .scaler import Scaler
from .temporal import add_temporal_features


class FeatureStats:
    """Accumulates mean/var/min/max without materializing all data."""

    def __init__(self, num_features):
        self.num_features = num_features
        self.count = np.zeros(num_features, dtype=np.float64)
        self.sum = np.zeros(num_features, dtype=np.float64)
        self.sum_sq = np.zeros(num_features, dtype=np.float64)
        self.min_value = np.full(num_features, np.inf, dtype=np.float64)
        self.max_value = np.full(num_features, -np.inf, dtype=np.float64)

    def update(self, array):
        if array.size == 0:
            return
        flat = array.reshape(-1, self.num_features)
        valid_mask = ~np.isnan(flat)
        if not valid_mask.any():
            return

        valid_counts = valid_mask.sum(axis=0)
        valid_values = np.where(valid_mask, flat, 0.0)
        self.count += valid_counts
        self.sum += valid_values.sum(axis=0)
        self.sum_sq += (valid_values**2).sum(axis=0)

        local_min = self.min_value.copy()
        local_max = self.max_value.copy()
        for idx in range(self.num_features):
            if valid_counts[idx] == 0:
                continue
            column = flat[:, idx]
            column_valid = column[valid_mask[:, idx]]
            local_min[idx] = column_valid.min()
            local_max[idx] = column_valid.max()

        self.min_value = np.minimum(self.min_value, local_min)
        self.max_value = np.maximum(self.max_value, local_max)

    def to_scaler(self, mode="min-max", eps=1e-5):
        scaler = Scaler(mode=mode, eps=eps)
        valid = self.count > 0

        if mode in {"standardization", "standardization_arcsinh"}:
            mean = np.zeros(self.num_features, dtype=np.float32)
            var = np.ones(self.num_features, dtype=np.float32)
            mean[valid] = (self.sum[valid] / self.count[valid]).astype(np.float32)
            var_values = (self.sum_sq[valid] / self.count[valid]) - (mean[valid] ** 2)
            var_values = np.where(var_values < 0, 0, var_values)
            var[valid] = var_values.astype(np.float32)
            scaler.mean = mean
            scaler.var = var
        elif mode == "min-max":
            min_val = np.zeros(self.num_features, dtype=np.float32)
            max_val = np.ones(self.num_features, dtype=np.float32)
            min_val[valid] = self.min_value[valid].astype(np.float32)
            max_val[valid] = self.max_value[valid].astype(np.float32)
            scaler.min_value = min_val
            scaler.max_value = max_val
        else:
            raise ValueError("Unsupported scaler mode")
        return scaler


def load_column_classes(legend_path):
    """Reads `csv_legend.xlsx` without requiring optional excel engines."""
    legend_path = Path(legend_path)
    if not legend_path.exists():
        return {}

    with zipfile.ZipFile(legend_path) as archive:
        strings_xml = archive.read("xl/sharedStrings.xml")
        sheet_xml = archive.read("xl/worksheets/sheet1.xml")

    strings_root = ET.fromstring(strings_xml)
    strings = [
        "".join(node.itertext())
        for node in strings_root.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si")
    ]

    sheet_root = ET.fromstring(sheet_xml)
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    rows = []
    for row in sheet_root.findall(".//main:row", ns):
        record = {}
        for cell in row.findall("main:c", ns):
            position = cell.get("r")
            column_letter = "".join(filter(str.isalpha, position))
            value_node = cell.find("main:v", ns)
            if value_node is None:
                continue
            value_text = value_node.text
            if cell.get("t") == "s":
                value_text = strings[int(value_text)]
            record[column_letter] = value_text
        if record:
            rows.append(record)

    column_classes = {}
    for entry in rows:
        name = entry.get("A")
        cls = entry.get("B")
        if name and cls and name != "Column name":
            column_classes[name] = cls.strip()
    return column_classes


class DatasetCacheBuilder:
    def __init__(
        self,
        data_root="Data/global/train",
        cache_root="Data/cache/train",
        metadata_path="Data/dataSummary_completed.csv",
        legend_path="Data/csv_legend.xlsx",
        target_column="avg_NDVI_clear_sky",
        history_window=5,
        forecast_window=5,
        step=6,
        noise_percentage=0.1,
        scaler_mode="min-max",
        add_temporal=True,
        fourier_harmonics=3,
        include_hour=False,
        include_day=False,
        include_month=False,
        seed=42,
        show_progress=True,
        group_mode="directory",
        pattern_index=2,
        min_samples_per_month=100,
        g5_target=2.0,
    ):
        self.data_root = Path(data_root)
        self.cache_root = Path(cache_root)
        self.metadata_path = Path(metadata_path)
        self.legend_path = Path(legend_path)
        self.target_column = target_column
        self.history_window = history_window
        self.forecast_window = forecast_window
        self.step = step
        self.noise_percentage = noise_percentage
        self.scaler_mode = scaler_mode
        self.add_temporal = add_temporal
        self.fourier_harmonics = fourier_harmonics
        self.include_hour = include_hour
        self.include_day = include_day
        self.include_month = include_month
        self.show_progress = show_progress
        self.group_mode = group_mode
        self.pattern_index = pattern_index

        self.min_samples_per_month = int(min_samples_per_month)
        self.g5_target = float(g5_target)

        self.rng = np.random.default_rng(seed)
        self.logger = logging.getLogger(self.__class__.__name__)

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_root}")
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self.column_classes = load_column_classes(self.legend_path)
        self.meteo_columns = [
            name for name, cls in self.column_classes.items() if cls.lower() == "meteo"
        ]
        if not self.meteo_columns:
            self.logger.warning("No meteo columns found in legend, fallback to all feature columns")

        self.metadata_by_cube, self.metadata_by_area = load_metadata_lookup(self.metadata_path)

        self.feature_names = None
        self.base_feature_columns = None
        self.stats_accumulator = None
        self.target_stats = None

        self.month_counts_area = defaultdict(lambda: defaultdict(int))
        self.month_counts_global = defaultdict(int)
        self.monthly_std_area = defaultdict(lambda: defaultdict(dict))
        self.monthly_std_global = defaultdict(dict)

    def progress(self, iterable, desc):
        if self.show_progress:
            return tqdm(iterable, desc=desc)
        return iterable

    def build(self):
        self.compute_monthly_stats()

        groups = self.collect_groups()
        if not groups:
            self.logger.warning("No CSV files found in %s", self.data_root)
            return

        area_names = sorted(groups.keys())
        for area_name in self.progress(area_names, "Areas"):
            self.process_area(area_name, groups[area_name])

        if self.stats_accumulator is not None and self.feature_names is not None:
            scaler = self.stats_accumulator.to_scaler(mode=self.scaler_mode)
            if self.target_stats is not None:
                scaler = self._attach_target_stats(scaler, self.target_stats)
            scaler_path = self.cache_root / "scaler.json"
            scaler.save(scaler_path)
            self.logger.info("Saved global scaler to %s", scaler_path)

    def collect_groups(self):
        groups = defaultdict(list)
        if self.group_mode == "directory":
            for area_dir in sorted(self.data_root.iterdir()):
                if area_dir.is_dir():
                    csv_files = sorted(area_dir.glob("*.csv"))
                    if csv_files:
                        groups[area_dir.name].extend(csv_files)
        elif self.group_mode == "pattern":
            for csv_path in self.data_root.rglob("*.csv"):
                if not csv_path.is_file():
                    continue
                parts = csv_path.stem.split("_")
                if len(parts) > self.pattern_index:
                    area_code = parts[self.pattern_index]
                    groups[area_code].append(csv_path)
        else:
            raise ValueError("group_mode must be 'directory' or 'pattern'")

        return {area: sorted(files) for area, files in groups.items() if files}

    def resolve_sample_metadata(self, csv_path: Path, area_name: str) -> SampleMetadata:
        by_cube = self.metadata_by_cube.get(csv_path.stem)
        area_default = self.metadata_by_area[area_name]
        if by_cube is None:
            return area_default

        climate = by_cube.climate if by_cube.climate != "unknown" else area_default.climate
        latitude = by_cube.latitude if np.isfinite(by_cube.latitude) else area_default.latitude
        longitude = by_cube.longitude if np.isfinite(by_cube.longitude) else area_default.longitude
        crop_type = by_cube.crop_type if by_cube.crop_type != "unknown" else area_default.crop_type
        return SampleMetadata(climate=climate, latitude=latitude, longitude=longitude, crop_type=crop_type)

    def process_area(self, area_name, csv_files):
        samples_history = []
        samples_history_mask = []
        samples_history_timestamps = []
        samples_future = []
        samples_future_mask = []
        samples_future_noise = []
        samples_future_timestamps = []
        samples_target = []
        samples_target_mask = []
        samples_target_timestamps = []
        samples_future_target_positions = []

        samples_climate = []
        samples_latitude = []
        samples_longitude = []
        samples_crop_type = []

        metadata_records = []

        area_defaults = self.metadata_by_area[area_name]

        for csv_path in self.progress(csv_files, f"{area_name}"):
            dataframe = self.load_csv(csv_path)
            if dataframe is None or dataframe.empty:
                continue

            feature_frame, target_series = self.prepare_features(dataframe)
            if feature_frame is None:
                continue

            columns = feature_frame.columns.tolist()
            history_target_name = f"{self.target_column}_history"
            if self.feature_names is None:
                self.base_feature_columns = columns
                self.feature_names = columns + [history_target_name]
                self.initialise_stats(len(self.feature_names))
            elif columns != self.base_feature_columns:
                self.logger.error("Column order mismatch in %s", csv_path)
                raise ValueError("Inconsistent columns across CSVs")

            sample_meta = self.resolve_sample_metadata(csv_path, area_name)
            climate_value = sample_meta.climate if sample_meta.climate != "unknown" else area_defaults.climate
            latitude_value = sample_meta.latitude
            longitude_value = sample_meta.longitude
            crop_type_value = sample_meta.crop_type

            feature_values = feature_frame.to_numpy(dtype=np.float32)
            feature_masks = np.isnan(feature_values)
            target_values = target_series.to_numpy(dtype=np.float32)
            timestamps = feature_frame.index.to_numpy()
            total_rows = len(feature_frame)

            valid_positions = [idx for idx in range(total_rows) if not math.isnan(target_values[idx])]
            if not valid_positions:
                continue

            if self.step <= 0:
                start_positions = valid_positions
            else:
                start_positions = [valid_positions[i] for i in range(0, len(valid_positions), self.step)]

            for start in start_positions:
                history_indices = []
                history_target_count = 0
                ptr = start

                while ptr < total_rows and history_target_count < self.history_window:
                    history_indices.append(ptr)
                    if not math.isnan(target_values[ptr]):
                        history_target_count += 1
                    ptr += 1

                if history_target_count < self.history_window:
                    continue

                history_slice = feature_values[history_indices]
                history_ts = timestamps[history_indices]

                target_history_column = np.full((len(history_indices), 1), np.nan, dtype=np.float32)
                for pos, original_idx in enumerate(history_indices):
                    value = target_values[original_idx]
                    if not math.isnan(value):
                        target_history_column[pos, 0] = value

                history_slice = np.concatenate([history_slice, target_history_column], axis=1)
                history_mask = np.isnan(history_slice)

                future_indices = []
                future_target_values = []
                future_target_timestamps = []
                future_target_positions = []
                while ptr < total_rows and len(future_target_values) < self.forecast_window:
                    future_indices.append(ptr)
                    if not math.isnan(target_values[ptr]):
                        future_target_values.append(target_values[ptr])
                        future_target_timestamps.append(str(timestamps[ptr]))
                        future_target_positions.append(len(future_indices) - 1)
                    ptr += 1

                if len(future_target_values) < self.forecast_window:
                    continue

                future_slice = feature_values[future_indices]
                future_mask = feature_masks[future_indices]
                future_ts = timestamps[future_indices]

                future_target_history_column = np.full((len(future_indices), 1), np.nan, dtype=np.float32)
                future_slice = np.concatenate([future_slice, future_target_history_column], axis=1)
                future_mask = np.concatenate([future_mask, np.isnan(future_target_history_column)], axis=1)

                target_slice = np.array(future_target_values, dtype=np.float32)
                target_mask = np.zeros_like(target_slice, dtype=bool)

                future_noisy, noise = self.add_noise_to_future(
                    future_slice,
                    future_mask,
                    future_timestamps=future_ts,
                    history_end_ts=history_ts[-1],
                    area_name=area_name,
                )

                if self.stats_accumulator is not None:
                    self.stats_accumulator.update(history_slice)
                    self.stats_accumulator.update(future_slice)
                    if self.target_stats is not None:
                        self.target_stats.update(target_slice.reshape(-1, 1))

                samples_history.append(history_slice)
                samples_history_mask.append(history_mask)
                samples_history_timestamps.append([str(ts) for ts in history_ts])
                samples_future.append(future_noisy)
                samples_future_mask.append(future_mask)
                samples_future_noise.append(noise)
                samples_future_timestamps.append([str(ts) for ts in future_ts])
                samples_target.append(target_slice)
                samples_target_mask.append(target_mask)
                samples_target_timestamps.append(future_target_timestamps)
                samples_future_target_positions.append(future_target_positions)

                samples_climate.append(climate_value)
                samples_latitude.append(latitude_value)
                samples_longitude.append(longitude_value)
                samples_crop_type.append(crop_type_value)

                metadata_records.append(
                    {
                        "area": area_name,
                        "csv": csv_path.name,
                        "history_start": samples_history_timestamps[-1][0],
                        "history_end": samples_history_timestamps[-1][-1],
                        "future_start": future_target_timestamps[0],
                        "future_end": future_target_timestamps[-1],
                        "climate": climate_value,
                        "latitude": None if not np.isfinite(latitude_value) else float(latitude_value),
                        "longitude": None if not np.isfinite(longitude_value) else float(longitude_value),
                        "crop_type": crop_type_value,
                    }
                )

        if not samples_history:
            self.logger.info("No valid samples for %s", area_name)
            return

        cache_path = self.cache_root / f"{area_name}.npz"
        np.savez_compressed(
            cache_path,
            history=np.array(samples_history, dtype=object),
            history_mask=np.array(samples_history_mask, dtype=object),
            history_timestamps=np.array(samples_history_timestamps, dtype=object),
            future=np.array(samples_future, dtype=object),
            future_mask=np.array(samples_future_mask, dtype=object),
            future_noise=np.array(samples_future_noise, dtype=object),
            future_timestamps=np.array(samples_future_timestamps, dtype=object),
            target=np.stack(samples_target).astype(np.float32),
            target_mask=np.stack(samples_target_mask).astype(bool),
            future_target_positions=np.array(samples_future_target_positions, dtype=np.int64),
            target_timestamps=np.array(samples_target_timestamps, dtype=object),
            climate=np.array(samples_climate, dtype=object),
            latitude=np.array(samples_latitude, dtype=np.float32),
            longitude=np.array(samples_longitude, dtype=np.float32),
            crop_type=np.array(samples_crop_type, dtype=object),
        )

        metadata = {
            "area": area_name,
            "feature_names": self.feature_names,
            "target_column": self.target_column,
            "history_window": self.history_window,
            "forecast_window": self.forecast_window,
            "step": self.step,
            "noise_percentage": self.noise_percentage,
            "scaler_mode": self.scaler_mode,
            "noise_mode": "monthly",
            "min_samples_per_month": self.min_samples_per_month,
            "g5_target": self.g5_target,
            "metadata_path": str(self.metadata_path),
            "metadata_records": metadata_records,
        }

        metadata_path = self.cache_root / f"{area_name}.json"
        with metadata_path.open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)

        self.logger.info("Created %d samples for %s", len(samples_history), area_name)

    def initialise_stats(self, num_features):
        if self.stats_accumulator is None:
            self.stats_accumulator = FeatureStats(num_features)
        if self.target_stats is None:
            self.target_stats = FeatureStats(1)

    def _attach_target_stats(self, scaler, target_stats: FeatureStats):
        if scaler.mode in {"standardization", "standardization_arcsinh"}:
            scaler.target_mean = target_stats.sum / np.maximum(target_stats.count, 1)
            var_values = (target_stats.sum_sq / np.maximum(target_stats.count, 1)) - (scaler.target_mean**2)
            var_values = np.where(var_values < 0, 0, var_values)
            scaler.target_var = var_values.astype(np.float32)
        elif scaler.mode == "min-max":
            scaler.target_min_value = target_stats.min_value.astype(np.float32)
            scaler.target_max_value = target_stats.max_value.astype(np.float32)
        return scaler

    def load_csv(self, path):
        try:
            df = pd.read_csv(path)
            base_seen = set()
            duplicate_columns = []
            for column in df.columns:
                base_name = column
                if "." in column:
                    root, suffix = column.rsplit(".", 1)
                    if suffix.isdigit():
                        base_name = root
                if base_name in base_seen:
                    duplicate_columns.append(column)
                else:
                    base_seen.add(base_name)

            if duplicate_columns:
                self.logger.debug(
                    "Duplicate columns in %s: %s. Keeping only first occurrence",
                    path,
                    duplicate_columns,
                )
                df = df.drop(columns=duplicate_columns)
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Error reading %s: %s", path, exc)
            return None

        if "time" not in df.columns:
            self.logger.warning("Missing time column in %s", path)
            return None

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates("time").set_index("time")

        if "wind" in df.columns:
            df["wind"] = pd.to_numeric(df["wind"], errors="coerce").fillna(0.0)

        numeric_columns = [col for col in df.columns if col != "wind"]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

        if self.add_temporal:
            df = add_temporal_features(
                df,
                add_hour=self.include_hour,
                add_day=self.include_day,
                add_month=self.include_month,
                add_day_of_year=True,
                fourier_harmonics=self.fourier_harmonics,
            )

        return df

    def prepare_features(self, dataframe):
        if self.target_column not in dataframe.columns:
            self.logger.warning("Target column %s missing", self.target_column)
            return None, None

        feature_columns = [col for col in dataframe.columns if col != self.target_column]
        drop_columns = [
            col
            for col in VEGETATION_COLUMNS
            if col != self.target_column and col in feature_columns
        ]
        feature_frame = dataframe.drop(columns=drop_columns)

        feature_columns = [col for col in feature_frame.columns if col != self.target_column]
        feature_frame = feature_frame[feature_columns]
        target_series = dataframe[self.target_column]
        return feature_frame, target_series

    def add_noise_to_future(self, future_slice, future_mask, future_timestamps=None, history_end_ts=None, area_name=None):
        return self._add_monthly_noise(
            future_slice,
            future_mask,
            future_timestamps=future_timestamps,
            history_end_ts=history_end_ts,
            area_name=area_name,
        )

    def _columns_to_perturb(self, num_features):
        columns_to_perturb = (
            [self.feature_names.index(name) for name in self.meteo_columns if name in self.feature_names]
            if self.feature_names
            else list(range(num_features))
        )
        if not columns_to_perturb:
            columns_to_perturb = list(range(num_features))
        return columns_to_perturb

    def compute_monthly_stats(self):
        groups = self.collect_groups()
        if not groups:
            self.logger.warning("No CSV files found in %s", self.data_root)
            return

        sums_area = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        sumsq_area = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        count_area = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        sums_global = defaultdict(lambda: defaultdict(float))
        sumsq_global = defaultdict(lambda: defaultdict(float))
        count_global = defaultdict(lambda: defaultdict(int))

        for area_name in self.progress(sorted(groups.keys()), "Monthly stats"):
            for csv_path in groups[area_name]:
                dataframe = self.load_csv(csv_path)
                if dataframe is None or dataframe.empty:
                    continue
                feature_frame, _ = self.prepare_features(dataframe)
                if feature_frame is None or feature_frame.empty:
                    continue

                meteo_cols = [c for c in self.meteo_columns if c in feature_frame.columns]
                if not meteo_cols:
                    meteo_cols = list(feature_frame.columns)

                meteo_frame = feature_frame[meteo_cols]
                months = meteo_frame.index.month
                for month in range(1, 13):
                    month_mask = months == month
                    if not month_mask.any():
                        continue
                    month_slice = meteo_frame.loc[month_mask]
                    self.month_counts_area[area_name][month] += len(month_slice)
                    self.month_counts_global[month] += len(month_slice)
                    for col in meteo_cols:
                        values = month_slice[col].to_numpy(dtype=np.float64, copy=False)
                        values = values[~np.isnan(values)]
                        if values.size == 0:
                            continue
                        sums_area[area_name][month][col] += float(values.sum())
                        sumsq_area[area_name][month][col] += float((values**2).sum())
                        count_area[area_name][month][col] += int(values.size)
                        sums_global[month][col] += float(values.sum())
                        sumsq_global[month][col] += float((values**2).sum())
                        count_global[month][col] += int(values.size)

        def finalize_std(sums, sumsq, count):
            stds = {}
            for col, cnt in count.items():
                if cnt <= 1:
                    stds[col] = 1.0
                    continue
                mean = sums[col] / cnt
                var = (sumsq[col] / cnt) - (mean**2)
                stds[col] = float(math.sqrt(max(var, 0.0)))
            return stds

        for area_name, months in count_area.items():
            for month, col_counts in months.items():
                stds = finalize_std(
                    sums_area[area_name][month],
                    sumsq_area[area_name][month],
                    col_counts,
                )
                self.monthly_std_area[area_name][month] = stds

        for month, col_counts in count_global.items():
            stds = finalize_std(sums_global[month], sumsq_global[month], col_counts)
            self.monthly_std_global[month] = stds

    def _get_monthly_std(self, area_name, month, feature_name):
        area_count = self.month_counts_area.get(area_name, {}).get(month, 0)
        if area_count >= self.min_samples_per_month:
            std = self.monthly_std_area.get(area_name, {}).get(month, {}).get(feature_name)
        else:
            std = None
        if std is None:
            std = self.monthly_std_global.get(month, {}).get(feature_name)
        if std is None or not np.isfinite(std) or std <= 0:
            std = 1.0
        return std

    def _add_monthly_noise(self, future_slice, future_mask, future_timestamps=None, history_end_ts=None, area_name=None):
        noisy = future_slice.copy()
        noise_matrix = np.zeros_like(future_slice, dtype=np.float32)

        if self.noise_percentage is None or self.noise_percentage == 0:
            return noisy, noise_matrix
        if future_timestamps is None or history_end_ts is None or area_name is None:
            raise ValueError("monthly noise requires future_timestamps, history_end_ts, and area_name")

        columns_to_perturb = self._columns_to_perturb(future_slice.shape[1])
        valid_mask = ~future_mask[:, columns_to_perturb]
        if not valid_mask.any():
            return noisy, noise_matrix

        column_values = future_slice[:, columns_to_perturb]
        feature_names = [self.feature_names[idx] for idx in columns_to_perturb]

        future_ts = np.array(future_timestamps, dtype="datetime64[ns]")
        history_end = np.datetime64(history_end_ts)
        delta_days = (future_ts - history_end).astype("timedelta64[D]").astype(np.float32)
        delta_days = np.maximum(delta_days, 0.0)

        delta_t5 = float(delta_days.max()) if delta_days.size else 0.0
        beta = (self.g5_target - 1.0) / delta_t5 if delta_t5 > 0 else 0.0
        growth = np.maximum(1.0 + beta * delta_days, 1.0)

        month_std_cache = {}
        for month in range(1, 13):
            month_std_cache[month] = np.array(
                [self._get_monthly_std(area_name, month, fname) for fname in feature_names],
                dtype=np.float32,
            )

        months = [pd.Timestamp(ts).month for ts in future_ts]
        std_matrix = np.vstack([month_std_cache[m] for m in months]).astype(np.float32)
        scale = std_matrix * self.noise_percentage * growth[:, None]

        noise = self.rng.normal(loc=0.0, scale=scale, size=column_values.shape).astype(np.float32)
        noise = np.where(valid_mask, noise, 0.0)
        noise_matrix[:, columns_to_perturb] = noise
        noisy[:, columns_to_perturb] = np.where(valid_mask, column_values + noise, column_values)

        return noisy, noise_matrix


def build_default_cache():
    builder = DatasetCacheBuilder()
    builder.build()


def parse_args():
    parser = argparse.ArgumentParser(description="Monthly-noise cache builder for temporal datasets")
    parser.add_argument("--data-root", type=str, default="Data/global/train")
    parser.add_argument("--cache-root", type=str, default="Data/cache/train")
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="Data/dataSummary_completed.csv",
        help="Path to metadata CSV (dataSummary_completed.csv preferred)",
    )
    parser.add_argument("--legend-path", type=str, default="Data/csv_legend.xlsx")
    parser.add_argument("--target-column", type=str, default="avg_NDVI_clear_sky")
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--forecast-window", type=int, default=5)
    parser.add_argument("--step", type=int, default=6)
    parser.add_argument("--noise-percentage", type=float, default=0.1)
    parser.add_argument(
        "--scaler-mode",
        type=str,
        default="min-max",
        choices=("min-max", "standardization", "standardization_arcsinh"),
    )
    parser.add_argument("--add-temporal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fourier-harmonics", type=int, default=3)
    parser.add_argument("--include-hour", action="store_true")
    parser.add_argument("--include-day", action="store_true")
    parser.add_argument("--include-month", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--group-mode",
        type=str,
        default="directory",
        choices=("directory", "pattern"),
    )
    parser.add_argument("--pattern-index", type=int, default=2)
    parser.add_argument(
        "--min-samples-per-month",
        type=int,
        default=100,
        help="Minimum monthly sample count to use area-local monthly std",
    )
    parser.add_argument(
        "--g5-target",
        type=float,
        default=2.0,
        help="Noise growth factor at furthest future step (monthly mode)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logger verbosity for builder messages",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s - %(message)s")
    kwargs = vars(args)
    kwargs.pop("log_level", None)
    builder = DatasetCacheBuilder(**kwargs)
    builder.build()
