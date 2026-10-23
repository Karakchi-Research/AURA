# AURA: AES Utility for Real-time Anomaly Detection

AURA is a Python framework for experimenting with anomaly detection during
simulated AES-128 block processing. Each model reads a binary dataset file,
extracts timing and/or simulated sensor features, detects injected anomalies,
and exports standardized metrics and predictions.

## Features

- AES-128 encryption timing simulation using PyCryptodome
- Optical and thermal sensor simulation
- Single-sensor and multi-sensor detection
- Random Forest machine-learning models
- Statistical threshold models
- Hybrid threshold-plus-ML models
- Standardized CSV summaries and prediction exports
- Excel reports containing predictions and summary metrics
- Support for `.mp4`, `.avi`, and `.ravi` dataset files

## Requirements

- Python 3.7 or newer
- Packages listed in [`requirements.txt`](./requirements.txt):
  - `pycryptodome`
  - `pandas`
  - `scikit-learn`

## Installation

```bash
git clone https://github.com/Karakchi-Research/AURA.git
cd AURA

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On Windows, activate the virtual environment with:

```powershell
.venv\Scripts\activate
```

## Running the Complete Benchmark

The video dataset is distributed separately because of its size. Download
`AURA_Dataset.tar.gz` from the project’s **Releases** page and place it in the
repository root. Extract it before using the tool:

```bash
tar -xzvf AURA_Dataset.tar.gz Dataset
```

This restores the `Dataset/` directory used by the models. Then run:

```bash
./run.sh
```

The runner:

1. Finds every `.mp4`, `.avi`, and `.ravi` file in `Dataset/`.
2. Runs all 13 refined models in `Models/` against each file.
3. Limits each model to `MAX_BLOCKS` (Default: 10,000 blocks).
4. Writes one log per model and dataset file to `logs/`.
5. Writes summaries, predictions, and Excel reports to `results/`.

The script can be launched from any working directory because it resolves the
project root relative to its own location.

## Running an Individual Model

Run a model directly from the project root:

```bash
python3 Models/multi_fused_hybrid.py "Dataset/Laser Welds.mp4"
```

Available models:

| Model | Detection approach | Features |
| --- | --- | --- |
| `only_optical_ml.py` | ML | Optical |
| `only_thermal_ml.py` | ML | Thermal |
| `optical_timing_ml.py` | ML | Optical, timing |
| `optical_thermal_ml.py` | ML | Optical, thermal |
| `thermal_timing_ml.py` | ML | Thermal, timing |
| `multi_fused_ml.py` | ML | Optical, thermal, timing |
| `only_optical_threshold.py` | Threshold | Optical |
| `optical_timing_threshold.py` | Threshold | Optical, timing |
| `optical_thermal_threshold.py` | Threshold | Optical, thermal |
| `thermal_timing_threshold.py` | Threshold | Thermal, timing |
| `multi_fused_threshold.py` | Threshold | Optical, thermal, timing |
| `only_timing_hybrid.py` | Hybrid | Timing |
| `multi_fused_hybrid.py` | Hybrid | Optical, thermal, timing |

Threshold and hybrid models use a `calculate_thresholds()` method to derive
baseline thresholds. ML models train a `RandomForestClassifier` on their
declared feature set. All models use the same 10,000-block processing limit
and shared summary helpers.

## Outputs

Model executions create a `results/` directory containing:

- `summary.csv`: master summary across model runs
- Model-specific summary CSV files
- Prediction CSV files
- Excel reports with `predictions` and `summary` worksheets

The `run.sh` script creates a `logs/` directory containing separate logs named
using the model and dataset, for example:

```text
logs/multi_fused_hybrid_Laser Welds.log
```

Generated `logs/` and `results/` directories are runtime output and are not
required to run an individual model.

## Project Layout

```text
AURA/
├── Dataset/                    # Input video and binary dataset files
├── Models/
│   ├── *.py                    # Refined detection models
│   ├── summary_helper.py       # Shared summary and console reporting
│   └── Deprecated/             # Legacy detectors retained for reference
├── README.md
├── requirements.txt
└── run.sh                      # Runs all refined models on all datasets
```

## Legacy Models

The files in `Models/Deprecated/` are retained for reference and compatibility.
They are not included in `run.sh`.

## Credits

**Developed by:** Nishant Chinnasami
**Advisor:** Dr. Rasha Karakchi
**Institution:** University of South Carolina

This work was supported under the McNair Junior Fellowship and Magellan Scholar
Program at the University of South Carolina.

Special thanks to **Rye Stahle-Smith** for hardware testing and experimental
support.
