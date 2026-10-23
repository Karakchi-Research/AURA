#!/bin/bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR" || exit 1

mkdir -p logs

models=(
  "Models/optical_thermal_ml.py"
  "Models/optical_thermal_threshold.py"
  "Models/optical_timing_ml.py"
  "Models/optical_timing_threshold.py"
  "Models/thermal_timing_ml.py"
  "Models/thermal_timing_threshold.py"
  "Models/only_timing_hybrid.py"
  "Models/only_optical_ml.py"
  "Models/only_optical_threshold.py"
  "Models/only_thermal_ml.py"
  "Models/multi_fused_ml.py"
  "Models/multi_fused_threshold.py"
  "Models/multi_fused_hybrid.py"
)

shopt -s nullglob
datasets=(Dataset/*.mp4 Dataset/*.avi Dataset/*.ravi)

for f in "${datasets[@]}"; do
  [ -f "$f" ] || continue
  dataset_name="$(basename "$f")"
  dataset_base="${dataset_name%.*}"
  echo "Running models on $dataset_name"

  for model in "${models[@]}"; do
    model_name="$(basename "$model" .py)"
    echo "  $model_name"
    python3 "$model" "$f" \
      > "logs/${model_name}_${dataset_base}.log" 2>&1
  done
done