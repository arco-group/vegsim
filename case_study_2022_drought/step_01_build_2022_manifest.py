#!/usr/bin/env python3
"""Build a sample manifest for windows intersecting JJA 2022.

Reads cache .npz files directly and writes a sample-level CSV with:
- split/cache root
- source file + sample index
- lat/lon
- country (France/Spain/other)
- window start/end timestamps
- whether the window intersects JJA 2022
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
    p = argparse.ArgumentParser(description="Build 2022 JJA sample manifest from cache roots.")
    p.add_argument("--cache-root", action="append", required=True, help="Repeatable cache directory")
    p.add_argument("--countries", nargs="+", default=["France", "Spain"])
    p.add_argument("--country-mode", choices=["bbox", "shapefile"], default="bbox")
    p.add_argument("--output-dir", default="case_study_2022_drought/outputs")
    return p.parse_args()


def _load_country_shapes():
    if gpd is None:
        return None
    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        # geopandas dataset may be unavailable in recent versions
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
    # Fast default: country-level bounding boxes.
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


def _split_name_from_cache(cache_root: Path) -> str:
    name = cache_root.name.lower()
    if "ood-st" in name:
        return "ood-st"
    if "ood-t" in name:
        return "ood-t"
    if "ood-s" in name:
        return "ood-s"
    if "val" in name:
        return "val"
    if "train" in name:
        return "train"
    return cache_root.name


def _to_datetime_list(obj) -> list[pd.Timestamp]:
    return [pd.to_datetime(str(x), errors="coerce") for x in obj]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world = _load_country_shapes() if args.country_mode == "shapefile" else None

    jja_start = pd.Timestamp("2022-06-01")
    jja_end = pd.Timestamp("2022-08-31 23:59:59")

    rows: list[dict] = []

    for cache_text in args.cache_root:
        cache_root = Path(cache_text)
        if not cache_root.exists():
            print(f"[WARN] Missing cache root: {cache_root}")
            continue
        split = _split_name_from_cache(cache_root)

        for npz_path in sorted(cache_root.glob("*.npz")):
            data = np.load(npz_path, allow_pickle=True)
            required = ["future_timestamps", "history_timestamps", "latitude", "longitude"]
            if any(k not in data.files for k in required):
                print(f"[WARN] Skip {npz_path.name}: missing required keys")
                continue

            n = len(data["future_timestamps"])
            lats = np.asarray(data["latitude"], dtype=np.float64)
            lons = np.asarray(data["longitude"], dtype=np.float64)

            for i in range(n):
                fut_ts = _to_datetime_list(data["future_timestamps"][i])
                hist_ts = _to_datetime_list(data["history_timestamps"][i])
                fut_ts = [t for t in fut_ts if pd.notna(t)]
                hist_ts = [t for t in hist_ts if pd.notna(t)]
                if not fut_ts and not hist_ts:
                    continue

                win_start = min((hist_ts + fut_ts))
                win_end = max((hist_ts + fut_ts))
                intersects_jja = not (win_end < jja_start or win_start > jja_end)
                future_has_jja = any((t >= jja_start) and (t <= jja_end) for t in fut_ts)

                lat = float(lats[i])
                lon = float(lons[i])
                country = _country_from_latlon(lat, lon, world, args.country_mode)

                rows.append(
                    {
                        "split": split,
                        "cache_root": str(cache_root),
                        "file": npz_path.name,
                        "sample_idx": i,
                        "lat": lat,
                        "lon": lon,
                        "country": country,
                        "window_start": str(win_start.date()),
                        "window_end": str(win_end.date()),
                        "future_start": str(min(fut_ts).date()) if fut_ts else "",
                        "future_end": str(max(fut_ts).date()) if fut_ts else "",
                        "intersects_jja_2022": bool(intersects_jja),
                        "future_has_jja_2022": bool(future_has_jja),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No samples found. Check cache roots.")

    df = df.sort_values(["country", "split", "file", "sample_idx"]).reset_index(drop=True)
    all_csv = out_dir / "manifest_2022_jja.csv"
    df.to_csv(all_csv, index=False)
    print(f"[OK] Saved {all_csv} ({len(df)} rows)")

    # Keep only windows where future actually includes JJA 2022.
    focus = df[df["future_has_jja_2022"]].copy()
    focus_csv = out_dir / "manifest_2022_jja_future_only.csv"
    focus.to_csv(focus_csv, index=False)
    print(f"[OK] Saved {focus_csv} ({len(focus)} rows)")

    countries = [str(c) for c in args.countries]
    for c in countries:
        sub = focus[focus["country"].str.lower() == c.lower()].copy()
        out = out_dir / f"manifest_2022_jja_{c.lower()}.csv"
        sub.to_csv(out, index=False)
        print(f"[OK] Saved {out} ({len(sub)} rows)")

    stats = (
        focus.groupby(["country", "split"], as_index=False)
        .size()
        .rename(columns={"size": "n_samples"})
        .sort_values(["country", "split"])
    )
    stats_csv = out_dir / "manifest_2022_jja_counts.csv"
    stats.to_csv(stats_csv, index=False)
    print(f"[OK] Saved {stats_csv}")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
