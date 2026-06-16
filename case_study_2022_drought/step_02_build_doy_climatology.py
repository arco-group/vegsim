#!/usr/bin/env python3
"""Build per-DOY climatology for temperature and precipitation (2017-2021).

The script scans cache .npz files and extracts:
- avg_temperature
- rainfall
from both history and future sequences, then aggregates medians by:
- country + day_of_year
- global day_of_year (fallback)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:  # pragma: no cover
    gpd = None
    Point = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DOY climatology from cache roots.")
    p.add_argument("--cache-root", action="append", required=True, help="Repeatable cache directory")
    p.add_argument("--countries", nargs="+", default=["France", "Spain"])
    p.add_argument("--country-mode", choices=["bbox", "shapefile"], default="bbox")
    p.add_argument("--start-year", type=int, default=2017)
    p.add_argument("--end-year", type=int, default=2021)
    p.add_argument("--output-dir", default="case_study_2022_drought/outputs")
    return p.parse_args()


def _load_country_shapes():
    if gpd is None:
        return None
    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        try:
            from cartopy.io import shapereader

            shp = shapereader.natural_earth(
                resolution="110m",
                category="cultural",
                name="admin_0_countries",
            )
            world = gpd.read_file(shp)
        except Exception:
            return None
    return world.to_crs(epsg=4326)


def _country_from_latlon(lat: float, lon: float, world_gdf, mode: str) -> str:
    if mode == "bbox" or world_gdf is None or Point is None:
        if 36.0 <= lat <= 44.8 and -9.8 <= lon <= 4.0:
            return "Spain"
        if 41.0 <= lat <= 51.6 and -5.6 <= lon <= 10.2:
            return "France"
        return "other"
    p = Point(float(lon), float(lat))
    hit = world_gdf[world_gdf.geometry.contains(p)]
    if hit.empty:
        return "other"
    name_col = "name" if "name" in hit.columns else hit.columns[0]
    return str(hit.iloc[0][name_col])


def _read_feature_names(cache_root: Path, npz_name: str) -> list[str]:
    meta_path = cache_root / npz_name.replace(".npz", ".json")
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text())
    names = meta.get("feature_names", [])
    return [str(x) for x in names]


def _extract_idx(feature_names: list[str], target: str) -> int:
    for i, n in enumerate(feature_names):
        if str(n).lower() == target.lower():
            return i
    raise ValueError(f"Feature '{target}' not found in feature_names")


def _collect_from_sequence(
    seq_arr: np.ndarray,
    ts_arr,
    temp_idx: int,
    rain_idx: int,
    *,
    country: str,
    year_min: int,
    year_max: int,
    out_rows: list[dict],
) -> None:
    if seq_arr is None:
        return
    seq = np.asarray(seq_arr, dtype=np.float32)
    if seq.ndim != 2:
        return
    ts_list = [pd.to_datetime(str(t), errors="coerce") for t in ts_arr]
    n = min(seq.shape[0], len(ts_list))
    for j in range(n):
        ts = ts_list[j]
        if pd.isna(ts):
            continue
        year = int(ts.year)
        if year < year_min or year > year_max:
            continue
        doy = int(ts.dayofyear)
        out_rows.append(
            {
                "country": country,
                "year": year,
                "doy": doy,
                "avg_temperature": float(seq[j, temp_idx]),
                "rainfall": float(seq[j, rain_idx]),
            }
        )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world = _load_country_shapes() if args.country_mode == "shapefile" else None
    focus_countries = {c.lower() for c in args.countries}
    rows: list[dict] = []

    for cache_text in args.cache_root:
        cache_root = Path(cache_text)
        if not cache_root.exists():
            print(f"[WARN] Missing cache root: {cache_root}")
            continue
        for npz_path in sorted(cache_root.glob("*.npz")):
            data = np.load(npz_path, allow_pickle=True)
            required = ["history", "future", "history_timestamps", "future_timestamps", "latitude", "longitude"]
            if any(k not in data.files for k in required):
                continue

            feature_names = _read_feature_names(cache_root, npz_path.name)
            if not feature_names:
                continue

            try:
                temp_idx = _extract_idx(feature_names, "avg_temperature")
                rain_idx = _extract_idx(feature_names, "rainfall")
            except ValueError:
                continue

            n = len(data["future"])
            lats = np.asarray(data["latitude"], dtype=np.float64)
            lons = np.asarray(data["longitude"], dtype=np.float64)
            for i in range(n):
                country = _country_from_latlon(float(lats[i]), float(lons[i]), world, args.country_mode)
                if country.lower() not in focus_countries:
                    continue

                _collect_from_sequence(
                    seq_arr=data["history"][i],
                    ts_arr=data["history_timestamps"][i],
                    temp_idx=temp_idx,
                    rain_idx=rain_idx,
                    country=country,
                    year_min=int(args.start_year),
                    year_max=int(args.end_year),
                    out_rows=rows,
                )
                _collect_from_sequence(
                    seq_arr=data["future"][i],
                    ts_arr=data["future_timestamps"][i],
                    temp_idx=temp_idx,
                    rain_idx=rain_idx,
                    country=country,
                    year_min=int(args.start_year),
                    year_max=int(args.end_year),
                    out_rows=rows,
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows collected for requested years/countries.")

    country_clim = (
        df.groupby(["country", "doy"], as_index=False)
        .agg(
            avg_temperature_median=("avg_temperature", "median"),
            avg_temperature_mean=("avg_temperature", "mean"),
            rainfall_median=("rainfall", "median"),
            rainfall_mean=("rainfall", "mean"),
            n=("doy", "size"),
        )
        .sort_values(["country", "doy"])
    )
    out_country = out_dir / f"doy_climatology_{args.start_year}_{args.end_year}_country.csv"
    country_clim.to_csv(out_country, index=False)
    print(f"[OK] Saved {out_country} ({len(country_clim)} rows)")

    global_clim = (
        df.groupby(["doy"], as_index=False)
        .agg(
            avg_temperature_median=("avg_temperature", "median"),
            rainfall_median=("rainfall", "median"),
            n=("doy", "size"),
        )
        .sort_values(["doy"])
    )
    out_global = out_dir / f"doy_climatology_{args.start_year}_{args.end_year}_global.csv"
    global_clim.to_csv(out_global, index=False)
    print(f"[OK] Saved {out_global} ({len(global_clim)} rows)")

    raw_out = out_dir / f"doy_climatology_{args.start_year}_{args.end_year}_raw.csv"
    df.to_csv(raw_out, index=False)
    print(f"[OK] Saved {raw_out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
