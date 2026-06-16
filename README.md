<div align="center">
<h1>VegSim</h1>
<h3>A Geospatial World Model for Scenario-Conditioned Vegetation Simulation</h3>

[Irene Iele](https://scholar.google.com/citations?user=srLH7lkAAAAJ&hl=it&oi=ao)<sup>1</sup>,
[Elena Mulero Ayllon](https://scholar.google.com/citations?user=-BOMvaUAAAAJ&hl=it&oi=ao)<sup>1</sup>,
[Paolo Soda](https://scholar.google.com/citations?user=E7rcYCQAAAAJ&hl=it&oi=ao)<sup>1,2</sup>,
[Matteo Tortora](https://matteotortora.github.io)<sup>3</sup>

<sup>1</sup>University Campus Bio-Medico of Rome,
<sup>2</sup>Umea University,
<sup>3</sup>University of Genoa

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
</div>

## Overview

VegSim is a PyTorch Lightning codebase for vegetation-index forecasting and counterfactual scenario simulation. It includes:

- cache construction from geospatial time-series CSV files;
- baseline quantile forecasting models;
- a modular vegetation world model with latent dynamics;
- scenario-conditioned inference for meteorological perturbations;
- evaluation utilities for trained checkpoints.

The import package is currently named `world_matnet` for compatibility with the research code, while the public project and repository are named VegSim.

## Repository Layout

```text
src/world_matnet/
  data/        Dataset cache builder, scaling, metadata, temporal utilities
  models/      Baseline quantile model and vegetation world model
  training/    DataModule, Lightning modules, training CLI
  scenario/    Scenario perturbation utilities
  configs/     YAML presets for baseline and world-model experiments
scripts/       Cache building, training, evaluation, and scenario simulation CLIs
experiments/   Reusable scenario configuration files
case_study_2022_drought/
               Scripts for the 2022 drought scenario simulations
tests/         Synthetic smoke tests
```

Generated data, checkpoints, logs, and outputs are intentionally ignored by git.

## Installation

Create an environment with Python 3.9 or newer, then install the repository in editable mode:

```bash
git clone https://github.com/arco-group/vegsim.git
cd vegsim
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Optional analysis dependencies:

```bash
pip install -e ".[analysis]"
```

## Data Format

The training pipeline expects a cache generated from input CSV time series. A typical raw-data layout is:

```text
Data/
  global/
    train/
      <area_or_region>/
        *.csv
  dataSummary_completed.csv
```

Each CSV should contain timestamped covariates and the target vegetation index column. The metadata CSV is used to attach region-level attributes such as climate zone, latitude, longitude, and crop type.

Build a cache with:

```bash
python scripts/build_cache.py \
  --data-root Data/global/train \
  --cache-root Data/cache/train_avg_NDVI_clear_sky \
  --target-column avg_NDVI_clear_sky \
  --group-mode directory \
  --metadata-path Data/dataSummary_completed.csv
```

The main batch tensors used by the world model are:

- `history`: `[B, T_hist, C_in]`
- `future`: `[B, L, C_in]`
- `target_dense`: `[B, L, C_y]`
- `target_dense_mask`: `[B, L, C_y]`, where `True` means missing supervision
- `future_delta_days`: `[B, L]`
- `spatial_cont`: continuous spatial metadata, usually latitude and longitude
- `spatial_cat`: categorical spatial metadata, usually climate and crop type

## Training

Train a baseline quantile forecaster:

```bash
python scripts/train_lightning.py \
  --model-type baseline_forecaster \
  --cache-root Data/cache/train_avg_NDVI_clear_sky \
  --quantiles 0.1,0.5,0.9 \
  --epochs 200 \
  --batch-size 128 \
  --logger-type csv \
  --experiment-name baseline_run
```

Train the GRU vegetation world model from a preset:

```bash
python scripts/train_lightning.py \
  --config src/world_matnet/configs/world_model_gru.yaml \
  --model-type vegetation_world_model \
  --cache-root Data/cache/train_avg_NDVI_clear_sky \
  --logger-type csv \
  --experiment-name vegsim_gru
```

For W&B logging, use `--logger-type wandb` or `--logger-type both` and set `--wandb-project`, `--wandb-entity`, and related CLI flags as needed.

## Evaluation

Evaluate a trained world-model checkpoint:

```bash
python scripts/evaluate_world_model.py \
  --checkpoint checkpoints/vegsim_gru_vegetation_world_model_gru_seed42/last.ckpt \
  --cache-root Data/cache/ood-st_chopped_avg_NDVI_clear_sky \
  --scaler-path Data/cache/train_avg_NDVI_clear_sky/scaler.json \
  --metrics-original-scale true \
  --output outputs/world_model_eval_oodst.json
```

The evaluator reports aggregate MAE, RMSE, pinball loss, calibration diagnostics, per-horizon metrics, and per-region metrics.

## Scenario Simulation

Run baseline and perturbed meteorological scenarios from a trained world-model checkpoint:

```bash
python scripts/predict_scenarios_lightning.py \
  --checkpoint checkpoints/vegsim_gru_vegetation_world_model_gru_seed42/last.ckpt \
  --cache-root Data/cache/ood-st_chopped_avg_NDVI_clear_sky \
  --scaler-path Data/cache/train_avg_NDVI_clear_sky/scaler.json \
  --scenarios experiments/paper_scenarios_v1.yaml \
  --output outputs/scenario_predictions_oodst.npz
```

The scenario output contains baseline quantiles, scenario quantiles, median deltas, optional risk scores, geospatial metadata, and scenario metadata.

## 2022 Drought Case Study

The `case_study_2022_drought/` directory contains the reproducible workflow used to build 2022 summer manifests, compute 2017-2021 day-of-year climatologies, and run the two scenario-conditioned simulations used in the paper:

- `step03_tempm4_rainp40_add2`: temperature -4C, rainfall x1.4, rainfall +2
- `step03_tempp4_rainm40_sub2`: temperature +4C, rainfall x0.6, rainfall -2

Start with:

```bash
python case_study_2022_drought/step_01_build_2022_manifest.py --help
python case_study_2022_drought/step_02_build_doy_climatology.py --help
bash case_study_2022_drought/run_2022_two_scenarios.sh
```

## Testing

Run the synthetic smoke tests:

```bash
pytest -q
```

These tests exercise variable-length batching, sparse future supervision, the vegetation world-model forward pass, finite training losses, and basic scenario perturbations.

## Citation

If you use VegSim in academic work, please cite the accompanying paper. The BibTeX entry will be added here after the final proceedings metadata is available.
