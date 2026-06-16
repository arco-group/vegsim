#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root:
#   bash case_study_2022_drought/run_2022_two_scenarios.sh

CKPT="${CKPT:-checkpoints/wm_gru_ab08_spatial_climate_harm/best.ckpt}"
SCALER="${SCALER:-Data/cache/train_avg_NDVI_clear_sky/scaler.json}"
CACHE_OOD_T="${CACHE_OOD_T:-Data/cache/ood-t_chopped_avg_NDVI_clear_sky}"
CACHE_OOD_ST="${CACHE_OOD_ST:-Data/cache/ood-st_chopped_avg_NDVI_clear_sky}"
COUNTRY_CLIM="${COUNTRY_CLIM:-case_study_2022_drought/outputs/doy_climatology_2017_2021_country.csv}"
GLOBAL_CLIM="${GLOBAL_CLIM:-case_study_2022_drought/outputs/doy_climatology_2017_2021_global.csv}"
DATA_SUMMARY="${DATA_SUMMARY:-Data/dataSummary_completed.csv}"

OUT_COOLWET="${OUT_COOLWET:-case_study_2022_drought/outputs/step03_tempm4_rainp40_add2}"
OUT_HOTDRY="${OUT_HOTDRY:-case_study_2022_drought/outputs/step03_tempp4_rainm40_sub2}"

COMMON_ARGS=(
  --checkpoint "$CKPT"
  --cache-root "$CACHE_OOD_T"
  --cache-root "$CACHE_OOD_ST"
  --scaler-path "$SCALER"
  --country-climatology-csv "$COUNTRY_CLIM"
  --global-climatology-csv "$GLOBAL_CLIM"
  --countries France Spain
  --simulation-strategy real_offset
  --apply-scaling true
  --feature-engineering true
  --min-crop-pixels 60
  --data-summary-path "$DATA_SUMMARY"
  --batch-size 64
  --num-workers 4
)

python case_study_2022_drought/step_03_infer_real_vs_normal.py \
  "${COMMON_ARGS[@]}" \
  --temp-offset-c -4.0 \
  --rain-multiplier 1.4 \
  --rain-additive 2.0 \
  --output-dir "$OUT_COOLWET"

python case_study_2022_drought/step_03_infer_real_vs_normal.py \
  "${COMMON_ARGS[@]}" \
  --temp-offset-c 4.0 \
  --rain-multiplier 0.6 \
  --rain-additive -2.0 \
  --output-dir "$OUT_HOTDRY"

echo "Scenario simulation complete:"
echo "  $OUT_COOLWET"
echo "  $OUT_HOTDRY"
