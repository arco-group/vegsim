"""Metadata helpers for climate/geo/crop enrichment."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

AREA_PATTERN = re.compile(r"(\d{2}[A-Z]{3})")


@dataclass(frozen=True)
class SampleMetadata:
    climate: str
    latitude: float
    longitude: float
    crop_type: str


def extract_area_code(value: str) -> str | None:
    if not value:
        return None
    match = AREA_PATTERN.search(value)
    return match.group(1) if match else None


def extract_group_key(cube_key: str) -> str | None:
    area = extract_area_code(cube_key)
    if area:
        return area
    parts = str(cube_key).split("_")
    if len(parts) >= 3 and parts[0].lower() == "minicube":
        return parts[2]
    if parts:
        return parts[0]
    return None


def _safe_str(value, default="unknown"):
    if value is None:
        return default
    value = str(value).strip()
    if not value or value.lower() == "nan":
        return default
    return value


def _safe_float(value, default=np.nan):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mode_or_unknown(series: pd.Series) -> str:
    clean = series.dropna().astype(str).str.strip()
    clean = clean[clean != ""]
    clean = clean[clean.str.lower() != "nan"]
    if clean.empty:
        return "unknown"
    mode = clean.mode()
    if mode.empty:
        return clean.iloc[0]
    return mode.iloc[0]


def _median_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.median())


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_cube_key(path_value: str) -> str:
    normalized = str(path_value).replace("\\", "/")
    return Path(normalized).stem


def load_metadata_lookup(metadata_path: str | Path):
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return {}, defaultdict(lambda: SampleMetadata("unknown", np.nan, np.nan, "unknown"))

    df = pd.read_csv(metadata_path)
    if "path_to_catalogue" not in df.columns:
        return {}, defaultdict(lambda: SampleMetadata("unknown", np.nan, np.nan, "unknown"))

    climate_col = _pick_column(df, ["climate"])
    crop_col = _pick_column(df, ["crop_type", "crop_dominant", "nome_coltivazione"])
    lat_col = _pick_column(df, ["latitude", "lat"])
    lon_col = _pick_column(df, ["longitude", "lon"])

    work = df.copy()
    work["cube_key"] = work["path_to_catalogue"].map(_normalize_cube_key)
    work["area"] = work["cube_key"].map(extract_group_key)

    if climate_col is None:
        work["_climate"] = "unknown"
    else:
        work["_climate"] = work[climate_col].map(_safe_str)

    if crop_col is None:
        work["_crop_type"] = "unknown"
    else:
        work["_crop_type"] = work[crop_col].map(_safe_str)

    if lat_col is None:
        work["_latitude"] = np.nan
    else:
        work["_latitude"] = pd.to_numeric(work[lat_col], errors="coerce")

    if lon_col is None:
        work["_longitude"] = np.nan
    else:
        work["_longitude"] = pd.to_numeric(work[lon_col], errors="coerce")

    area_map = defaultdict(lambda: SampleMetadata("unknown", np.nan, np.nan, "unknown"))
    for area, grp in work.dropna(subset=["area"]).groupby("area"):
        area_map[area] = SampleMetadata(
            climate=_mode_or_unknown(grp["_climate"]),
            latitude=_median_or_nan(grp["_latitude"]),
            longitude=_median_or_nan(grp["_longitude"]),
            crop_type=_mode_or_unknown(grp["_crop_type"]),
        )

    by_cube = {}
    best_score = {}
    for row in work.itertuples(index=False):
        cube_key = getattr(row, "cube_key", "")
        if not cube_key:
            continue

        climate = _safe_str(getattr(row, "_climate", "unknown"))
        crop_type = _safe_str(getattr(row, "_crop_type", "unknown"))
        latitude = _safe_float(getattr(row, "_latitude", np.nan))
        longitude = _safe_float(getattr(row, "_longitude", np.nan))

        score = 0
        score += 1 if climate != "unknown" else 0
        score += 1 if crop_type != "unknown" else 0
        score += 1 if np.isfinite(latitude) else 0
        score += 1 if np.isfinite(longitude) else 0

        current_score = best_score.get(cube_key, -1)
        if score >= current_score:
            best_score[cube_key] = score
            by_cube[cube_key] = SampleMetadata(
                climate=climate,
                latitude=latitude,
                longitude=longitude,
                crop_type=crop_type,
            )

    return by_cube, area_map
