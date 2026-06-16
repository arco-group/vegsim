# 2022 Drought Scenario Simulation

This folder contains the scenario-conditioned simulation pipeline for the 2022 drought analysis:

- factual run: `pre-summer 2022 vegetation state + real meteo 2022`
- cool/wet counterfactual: `temperature -4C, rainfall x1.4, rainfall +2`
- hot/dry counterfactual: `temperature +4C, rainfall x0.6, rainfall -2`

Initial focus countries:

- France
- Spain

## Step 1: Build 2022 JJA manifest from OOD subsets

Create a sample-level manifest from cache samples and keep windows intersecting JJA 2022.

```bash
python case_study_2022_drought/step_01_build_2022_manifest.py \
  --cache-root Data/cache/ood-st_chopped_avg_NDVI_clear_sky \
  --cache-root Data/cache/ood-t_chopped_avg_NDVI_clear_sky \
  --countries France Spain \
  --output-dir case_study_2022_drought/outputs
```

Outputs:

- `case_study_2022_drought/outputs/manifest_2022_jja.csv`
- `case_study_2022_drought/outputs/manifest_2022_jja_france.csv`
- `case_study_2022_drought/outputs/manifest_2022_jja_spain.csv`

## Step 2: Build DOY climatology (2017-2021)

Estimate per-country median temperature/precipitation for each day-of-year.

```bash
python case_study_2022_drought/step_02_build_doy_climatology.py \
  --cache-root Data/cache/train_avg_NDVI_clear_sky \
  --cache-root Data/cache/val_chopped_avg_NDVI_clear_sky \
  --cache-root Data/cache/ood-s_chopped_avg_NDVI_clear_sky \
  --cache-root Data/cache/ood-t_chopped_avg_NDVI_clear_sky \
  --cache-root Data/cache/ood-st_chopped_avg_NDVI_clear_sky \
  --countries France Spain \
  --start-year 2017 \
  --end-year 2021 \
  --output-dir case_study_2022_drought/outputs
```

Outputs:

- `case_study_2022_drought/outputs/doy_climatology_2017_2021_country.csv`
- `case_study_2022_drought/outputs/doy_climatology_2017_2021_global.csv`

## Step 3: Run the two scenario-conditioned simulations

The repository includes the trained checkpoint expected by the launcher:

```text
checkpoints/wm_gru_ab08_spatial_climate_harm/best.ckpt
checkpoints/wm_gru_ab08_spatial_climate_harm/resolved_config.yaml
```

Run:

```bash
bash case_study_2022_drought/run_2022_two_scenarios.sh
```

This produces:

- `case_study_2022_drought/outputs/step03_tempm4_rainp40_add2`
- `case_study_2022_drought/outputs/step03_tempp4_rainm40_sub2`

The output directories contain factual predictions, scenario predictions, sample-level deltas, country summaries, and the simulation configuration used for each run.
