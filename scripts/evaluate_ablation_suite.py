#!/usr/bin/env python3
"""Batch evaluation utility for ablation checkpoints.

Runs ``scripts/evaluate_world_model.py`` for each discovered best checkpoint
and for each provided cache split, then aggregates overall metrics into tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all ablation checkpoints across multiple cache splits")
    parser.add_argument("--checkpoints-root", type=str, default="checkpoints/ablations_gru")
    parser.add_argument(
        "--checkpoint-glob",
        type=str,
        default="best-*.ckpt",
        help="Glob pattern used inside each run directory to pick best checkpoint(s)",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        action="append",
        required=True,
        help="Cache split root; repeat this flag for each split",
    )
    parser.add_argument(
        "--cache-alias",
        type=str,
        action="append",
        default=None,
        help="Alias per cache split (same order as --cache-root). If omitted, folder name is used.",
    )
    parser.add_argument("--scaler-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--apply-scaling", type=str2bool, default=True)
    parser.add_argument("--feature-engineering", type=str2bool, default=True)
    parser.add_argument("--min-crop-pixels", type=float, default=60.0)
    parser.add_argument("--data-summary-path", type=str, default="Data/dataSummary_completed.csv")
    parser.add_argument("--pin-memory", type=str2bool, default=False)
    parser.add_argument("--metrics-original-scale", type=str2bool, default=False)
    parser.add_argument(
        "--metrics-style",
        type=str,
        default="current",
        choices=["current", "agrimatnet_v2"],
        help="Metric computation style forwarded to evaluate_world_model.py",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/eval_ablations_gru")
    parser.add_argument("--skip-existing", type=str2bool, default=True)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means all runs")
    parser.add_argument("--dry-run", type=str2bool, default=False)
    parser.add_argument("--python-exec", type=str, default=sys.executable)
    return parser.parse_args()


def _sanitize(name: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name).strip())
    return text.strip("_") or "split"


def _run_label_from_dir(run_dir_name: str) -> str:
    # Example: wm_gru_ab03_horizon_sinusoidal_wm_gru_seed42 -> wm_gru_ab03_horizon_sinusoidal
    return re.sub(r"_wm_gru_seed\d+$", "", run_dir_name)


def discover_best_checkpoints(checkpoints_root: Path, checkpoint_glob: str) -> list[dict[str, str]]:
    if not checkpoints_root.exists():
        raise FileNotFoundError(f"Checkpoints root not found: {checkpoints_root}")

    runs: list[dict[str, str]] = []
    for run_dir in sorted(checkpoints_root.iterdir()):
        if not run_dir.is_dir():
            continue
        candidates = sorted(run_dir.glob(checkpoint_glob))
        if not candidates:
            continue
        # If more than one exists, pick most recent by mtime.
        best_ckpt = max(candidates, key=lambda p: p.stat().st_mtime)
        runs.append(
            {
                "run_dir": run_dir.name,
                "run_label": _run_label_from_dir(run_dir.name),
                "checkpoint": str(best_ckpt),
            }
        )
    return runs


def _build_eval_cmd(
    *,
    python_exec: str,
    evaluator_script: Path,
    checkpoint: str,
    cache_root: str,
    output_json: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        python_exec,
        str(evaluator_script),
        "--checkpoint",
        checkpoint,
        "--cache-root",
        cache_root,
        "--output",
        str(output_json),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--apply-scaling",
        str(bool(args.apply_scaling)).lower(),
        "--feature-engineering",
        str(bool(args.feature_engineering)).lower(),
        "--min-crop-pixels",
        str(float(args.min_crop_pixels)),
        "--data-summary-path",
        str(args.data_summary_path),
        "--pin-memory",
        str(bool(args.pin_memory)).lower(),
        "--metrics-original-scale",
        str(bool(args.metrics_original_scale)).lower(),
        "--metrics-style",
        str(args.metrics_style),
    ]
    if args.scaler_path:
        cmd.extend(["--scaler-path", str(args.scaler_path)])
    return cmd


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        return f"{value:.6f}"
    return str(value)


def _write_markdown_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("| " + " | ".join(columns) + " |\n")
        fp.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            fp.write("| " + " | ".join(_format_cell(row.get(col, "")) for col in columns) + " |\n")


def _to_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _pivot_metric(rows: list[dict[str, Any]], metric: str, cache_aliases: list[str]) -> list[dict[str, Any]]:
    by_run: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_label = str(row["run_label"])
        run_dir = str(row["run_dir"])
        ckpt = str(row["checkpoint"])
        alias = str(row["cache_alias"])
        val = _to_float(row.get(metric))
        rec = by_run.setdefault(
            run_label,
            {
                "run_label": run_label,
                "run_dir": run_dir,
                "checkpoint": ckpt,
            },
        )
        rec[alias] = val

    out = []
    for run_label, rec in sorted(by_run.items(), key=lambda kv: kv[0]):
        values = [_to_float(rec.get(alias)) for alias in cache_aliases if alias in rec]
        if values:
            rec["mean"] = sum(values) / len(values)
        else:
            rec["mean"] = float("nan")
        out.append(rec)
    return out


def _build_cache_aliases(cache_roots: list[str], cache_aliases: list[str] | None) -> list[str]:
    if not cache_aliases:
        return [_sanitize(Path(c).name) for c in cache_roots]
    if len(cache_aliases) != len(cache_roots):
        raise ValueError("--cache-alias must have same count as --cache-root")
    return [_sanitize(alias) for alias in cache_aliases]


def _is_nan(x: Any) -> bool:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return True
    return val != val


def _rank_values(values_by_run: dict[str, Any], *, higher_is_better: bool) -> dict[str, float]:
    valid_items: list[tuple[str, float]] = []
    for run_label, raw_val in values_by_run.items():
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            continue
        if val != val:  # NaN
            continue
        valid_items.append((run_label, val))

    if higher_is_better:
        valid_items.sort(key=lambda x: x[1], reverse=True)
    else:
        valid_items.sort(key=lambda x: x[1])

    ranks: dict[str, float] = {run_label: float("nan") for run_label in values_by_run}
    for idx, (run_label, _) in enumerate(valid_items, start=1):
        ranks[run_label] = float(idx)
    return ranks


def main():
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    evaluator_script = root / "scripts" / "evaluate_world_model.py"
    if not evaluator_script.exists():
        raise FileNotFoundError(f"Evaluator script not found: {evaluator_script}")

    checkpoints_root = Path(args.checkpoints_root)
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_json"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cache_roots = [str(Path(c)) for c in args.cache_root]
    cache_aliases = _build_cache_aliases(cache_roots, args.cache_alias)

    runs = discover_best_checkpoints(checkpoints_root, args.checkpoint_glob)
    if not runs:
        raise RuntimeError(f"No checkpoints discovered under {checkpoints_root} with pattern {args.checkpoint_glob}")
    if args.max_runs and args.max_runs > 0:
        runs = runs[: args.max_runs]

    print(f"[INFO] Discovered {len(runs)} run directories")
    print(f"[INFO] Evaluating on {len(cache_roots)} cache splits")

    rows: list[dict[str, Any]] = []
    for run in runs:
        run_dir = run["run_dir"]
        run_label = run["run_label"]
        ckpt = run["checkpoint"]

        for cache_root, cache_alias in zip(cache_roots, cache_aliases):
            out_json = raw_dir / f"{_sanitize(run_dir)}__{_sanitize(cache_alias)}.json"

            cmd = _build_eval_cmd(
                python_exec=args.python_exec,
                evaluator_script=evaluator_script,
                checkpoint=ckpt,
                cache_root=cache_root,
                output_json=out_json,
                args=args,
            )

            print(f"[EVAL] run={run_label} split={cache_alias}")
            if args.dry_run:
                print("[DRY] " + " ".join(cmd))
                continue

            if out_json.exists() and args.skip_existing:
                print(f"[SKIP] existing {out_json}")
            else:
                subprocess.run(cmd, check=True)

            with out_json.open("r", encoding="utf-8") as fp:
                metrics = json.load(fp)
            overall = metrics.get("overall", {})

            rows.append(
                {
                    "run_dir": run_dir,
                    "run_label": run_label,
                    "checkpoint": ckpt,
                    "cache_alias": cache_alias,
                    "cache_root": cache_root,
                    "mae": _to_float(overall.get("mae")),
                    "rmse": _to_float(overall.get("rmse")),
                    "weighted_pinball": _to_float(overall.get("weighted_pinball")),
                    "mae_mean": _to_float(overall.get("mae_mean")),
                    "mae_std": _to_float(overall.get("mae_std")),
                    "rmse_mean": _to_float(overall.get("rmse_mean")),
                    "rmse_std": _to_float(overall.get("rmse_std")),
                    "weighted_pinball_mean": _to_float(overall.get("weighted_pinball_mean")),
                    "weighted_pinball_std": _to_float(overall.get("weighted_pinball_std")),
                    "coverage": _to_float(overall.get("coverage")),
                    "interval_width": _to_float(overall.get("interval_width")),
                    "calibration_error": _to_float(overall.get("calibration_error")),
                    "wmape_mean": _to_float(overall.get("wmape_mean")),
                    "wmape_std": _to_float(overall.get("wmape_std")),
                    "mase_mean": _to_float(overall.get("mase_mean")),
                    "mase_std": _to_float(overall.get("mase_std")),
                    "crps_mean": _to_float(overall.get("crps_mean")),
                    "crps_std": _to_float(overall.get("crps_std")),
                }
            )

    if args.dry_run:
        print("[INFO] Dry-run completed; no files aggregated.")
        return

    if not rows:
        raise RuntimeError("No evaluation rows collected")

    long_fieldnames = [
        "run_dir",
        "run_label",
        "checkpoint",
        "cache_alias",
        "cache_root",
        "mae",
        "rmse",
        "weighted_pinball",
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "rmse_std",
        "weighted_pinball_mean",
        "weighted_pinball_std",
        "coverage",
        "interval_width",
        "calibration_error",
        "wmape_mean",
        "wmape_std",
        "mase_mean",
        "mase_std",
        "crps_mean",
        "crps_std",
    ]
    _write_csv(output_dir / "ablation_eval_long.csv", rows, long_fieldnames)
    with (output_dir / "ablation_eval_long.json").open("w", encoding="utf-8") as fp:
        json.dump(rows, fp, indent=2)

    # Pivot tables by metric
    pivot_metrics = [
        "weighted_pinball",
        "weighted_pinball_mean",
        "weighted_pinball_std",
        "rmse",
        "rmse_mean",
        "rmse_std",
        "mae",
        "mae_mean",
        "mae_std",
        "calibration_error",
        "coverage",
        "interval_width",
        "wmape_mean",
        "wmape_std",
        "mase_mean",
        "mase_std",
        "crps_mean",
        "crps_std",
    ]
    pivot_rows_by_metric: dict[str, list[dict[str, Any]]] = {}
    for metric in pivot_metrics:
        pivot_rows = _pivot_metric(rows, metric, cache_aliases)
        pivot_rows_by_metric[metric] = pivot_rows
        # Best = lower for all except coverage (higher). Keep simple CSV and leave ranking to downstream.
        metric_fieldnames = ["run_label", "run_dir", "checkpoint"] + cache_aliases + ["mean"]
        _write_csv(output_dir / f"ablation_eval_pivot_{metric}.csv", pivot_rows, metric_fieldnames)
        _write_markdown_table(output_dir / f"ablation_eval_pivot_{metric}.md", pivot_rows, metric_fieldnames)

    # Basic summary ranking on weighted pinball mean
    pin_rows = _pivot_metric(rows, "weighted_pinball", cache_aliases)
    pin_rows_sorted = sorted(pin_rows, key=lambda r: _to_float(r.get("mean"), default=float("inf")))
    rank_rows = []
    for rank, rec in enumerate(pin_rows_sorted, start=1):
        rank_rows.append(
            {
                "rank": rank,
                "run_label": rec["run_label"],
                "run_dir": rec["run_dir"],
                "mean_weighted_pinball": rec.get("mean"),
            }
        )
    _write_csv(output_dir / "ablation_eval_rank_weighted_pinball.csv", rank_rows, ["rank", "run_label", "run_dir", "mean_weighted_pinball"])
    _write_markdown_table(
        output_dir / "ablation_eval_rank_weighted_pinball.md",
        rank_rows,
        ["rank", "run_label", "run_dir", "mean_weighted_pinball"],
    )

    # Build master table (wide format + per-metric mean ranks), same structure used in model ablations.
    run_index: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_label = str(row["run_label"])
        rec = run_index.setdefault(
            run_label,
            {
                "run_label": run_label,
                "run_dir": row["run_dir"],
                "checkpoint": row["checkpoint"],
            },
        )
        # Keep first-seen run metadata.
        if not rec.get("run_dir"):
            rec["run_dir"] = row["run_dir"]
        if not rec.get("checkpoint"):
            rec["checkpoint"] = row["checkpoint"]

    for metric in pivot_metrics:
        for prow in pivot_rows_by_metric.get(metric, []):
            run_label = str(prow["run_label"])
            rec = run_index.setdefault(
                run_label,
                {
                    "run_label": run_label,
                    "run_dir": prow.get("run_dir"),
                    "checkpoint": prow.get("checkpoint"),
                },
            )
            for alias in cache_aliases:
                rec[f"{metric}__{alias}"] = _to_float(prow.get(alias))
            rec[f"{metric}__mean"] = _to_float(prow.get("mean"))

    # Rank means for each metric (coverage higher is better, others lower is better).
    metric_higher_is_better = {
        "weighted_pinball": False,
        "weighted_pinball_mean": False,
        "weighted_pinball_std": False,
        "rmse": False,
        "rmse_mean": False,
        "rmse_std": False,
        "mae": False,
        "mae_mean": False,
        "mae_std": False,
        "calibration_error": False,
        "coverage": True,
        "interval_width": False,
        "wmape_mean": False,
        "wmape_std": False,
        "mase_mean": False,
        "mase_std": False,
        "crps_mean": False,
        "crps_std": False,
    }
    for metric in pivot_metrics:
        values_by_run = {
            run_label: rec.get(f"{metric}__mean")
            for run_label, rec in run_index.items()
        }
        ranks = _rank_values(values_by_run, higher_is_better=metric_higher_is_better[metric])
        for run_label, rank in ranks.items():
            run_index[run_label][f"rank_{metric}_mean"] = rank

    def _sort_key(rec: dict[str, Any]) -> tuple[float, str]:
        rank = rec.get("rank_weighted_pinball_mean")
        if _is_nan(rank):
            return (float("inf"), str(rec.get("run_label", "")))
        return (float(rank), str(rec.get("run_label", "")))

    master_rows = sorted(run_index.values(), key=_sort_key)

    master_fieldnames = ["run_label", "run_dir", "checkpoint"]
    for metric in pivot_metrics:
        master_fieldnames.extend([f"{metric}__{alias}" for alias in cache_aliases])
        master_fieldnames.append(f"{metric}__mean")
    # Keep rank columns in sync with pivot_metrics automatically.
    master_fieldnames.extend([f"rank_{metric}_mean" for metric in pivot_metrics])
    _write_csv(output_dir / "ablation_eval_master.csv", master_rows, master_fieldnames)

    master_md_columns = [
        "run_label",
        "weighted_pinball__mean",
        "rank_weighted_pinball_mean",
        "rmse__mean",
        "mae__mean",
        "calibration_error__mean",
        "coverage__mean",
        "interval_width__mean",
        "wmape_mean__mean",
        "wmape_std__mean",
        "mase_mean__mean",
        "mase_std__mean",
        "crps_mean__mean",
        "crps_std__mean",
    ]
    _write_markdown_table(output_dir / "ablation_eval_master.md", master_rows, master_md_columns)

    print(f"[DONE] Saved aggregated outputs in: {output_dir}")


if __name__ == "__main__":
    main()
