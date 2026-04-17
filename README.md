# **AURA: Additive Manufacturing Utility for Real-time Anomaly Detection**
### *Framework for LPBF Thin-Wall Geometric Monitoring*

---

## 📌 Project Overview

**AURA** (*Additive Manufacturing Utility for Real-time Anomaly Detection*, evolved from its initial purpose as an *AES Utility for Real-time Anomaly Detection*) is a lightweight, modular framework for detecting geometric anomalies in Laser Powder Bed Fusion (LPBF) thin-wall structures through multi-modal optical and temporal analysis. The framework monitors 316L stainless steel thin walls (0.2–1.0 mm) during manufacturing using layer-wise optical imaging for in-situ wall thickness, surface roughness, and center drift quantification [1].

A key innovation is repurposing "timing" models originally designed for AES-128 side-channel analysis to analyze **temporal dynamics** (layer-to-layer evolution), enabling robust detection of manufacturing instabilities. The framework supports:

- **Spatial detection** — Wall thickness deviation, surface roughness, and center drift from individual layer images
- **Temporal detection** — Layer-to-layer cumulative changes indicating process instability
- **Hybrid detection** — Fusion of spatial + temporal features for improved robustness

The framework implements three primary detection strategies:
- **Threshold-based detection** — Statistical anomaly identification using adaptive thresholds on geometric features
- **ML-based detection** — Random Forest classifiers trained on engineered geometric features
- **Hybrid detection** — Fusion of both spatial and temporal detection approaches

---

## ⚙️ Features

- 🔍 **LPBF In-Situ Monitoring** — Layer-by-layer geometric tracking during thin-wall fabrication based on reference [1]; adapted from original AES-128 side-channel monitoring
- 📊 **Multi-Modal Feature Extraction** (per layer image):
  - **Spatial features**: Wall thickness (px), center X position, left/right edge detection, surface roughness (Laplacian), texture variance, contour irregularity
  - **Temporal features**: Layer-to-layer thickness deviation (%), center drift (px), rolling statistics, accumulation metrics, thickness velocity (evolved from AES execution timing patterns)
  - **Hybrid vectors**: Combined spatial + temporal for fused detection
- 👁️ **Computer Vision Pipeline**:
  - Gaussian blur + horizontal gradient-based edge detection (per reference [1])
  - Intensity profile averaging for robust wall boundary identification
  - Multi-scale roughness analysis via Laplacian operator
- 🎯 **Three Detection Modes** (originally developed for AES timing analysis, adapted for LPBF):
  - **Spatial Detection** — Threshold-based anomaly detection on wall geometry
  - **Temporal Detection** — Threshold-based anomaly detection on layer-to-layer changes  
  - **Hybrid Detection** — Fusion of spatial + temporal criteria for improved recall
- 🤖 **ML-Based Variants** — Random Forest classifiers for all three modes + adaptive thresholding (maintains original AES model architecture)
- 📷 **Batch Image Processing** — Efficient frame-by-frame feature extraction from layer image directories
- 📈 **Automated Reporting** — Excel (.xlsx) export with detailed geometric metrics, anomaly flags, and statistics
- 🏃 **Lightweight & Fast** — ~100ms per layer inference, suitable for real-time manufacturing feedback

---

## 📂 Repository Structure

```
AURA/
├── README.md                    # Project documentation
├── Requirements.txt             # Python dependencies
├── ML_Detect.py                 # [Legacy] AES-based ML detector (kept for compatibility)
├── Threshold_Detect.py          # [Legacy] AES-based threshold detector (kept for compatibility)
├── thermal_adapter.py           # [Legacy] Signal adapter (kept for compatibility)
│
├── data/                        # LPBF layer image datasets
│   ├── first layer/             # Optical images from Layer 1 (42 frames)
│   ├── second + third/          # Optical images from Layers 2-3
│   └── layer1.png - layer7.png  # Individual layer snapshots
│
└── Testing/                     # LPBF geometric monitoring test suite
    ├── _utils.py                # Shared utilities: image loading, geometric feature extraction per [1]
    ├── run_all.py               # Orchestrator for running all 6 test variants
    │
    ├── spatial_threshold.py      # [=optical_threshold.py] Spatial geometric anomaly detection
    ├── spatial_ml.py            # [=optical_ml.py] Spatial ML anomaly detection
    │
    ├── temporal_threshold.py     # [=timing_threshold.py] Temporal (layer-to-layer) anomaly detection
    ├── temporal_ml.py           # [=timing_ml.py] Temporal ML anomaly detection
    │
    ├── hybrid_threshold.py       # [=timing_optical_threshold.py] Spatial+temporal threshold fusion
    └── hybrid_ml.py             # [=timing_optical_ml.py] Spatial+temporal ML fusion
```

**Note**: File naming convention (`optical_*` / `timing_*`) is retained for backward compatibility with prior ablation study naming. Semantically:
- `optical_*` files = **Spatial** detection (wall geometry from images)
- `timing_*` files = **Temporal** detection (layer-to-layer dynamics)
- `timing_optical_*` files = **Hybrid** detection (spatial + temporal fusion)

### Core Framework Files (Root Directory)

The three primary files maintain backward compatibility with AES framework:

| File | Status | Purpose |
|------|--------|----------|
| **ML_Detect.py** | Legacy | AES-based ML anomaly classifier (replaced by Testing/spatial_ml.py for LPBF) |
| **Threshold_Detect.py** | Legacy | AES-based threshold detector (replaced by Testing/spatial_threshold.py for LPBF) |
| **thermal_adapter.py** | Legacy | Signal adapter (still usable with Testing models) |

For **LPBF thin-wall monitoring**, use the Testing suite directly.

### Testing Suite (Testing/ Directory)

Six LPBF geometric monitoring test scripts organized by detection strategy:

| Detection Strategy | Threshold-Based | ML-Based |
|--------|---|---|
| **Spatial Only** (from layer images) | `optical_threshold.py` | `optical_ml.py` |
| **Temporal Only** (layer-to-layer metrics) | `timing_threshold.py` | `timing_ml.py` |
| **Hybrid** (spatial + temporal fusion) | `timing_optical_threshold.py` | `timing_optical_ml.py` |

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.7+**
- **Virtual environment** (recommended: `venv` or `conda`)
- **Dependencies**: `pandas`, `numpy`, `scikit-learn`, `opencv-python`, `openpyxl`, `psutil`
  - Note: `pycryptodome` is optional (legacy AES framework only)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nishant640/AURA.git
   cd AURA
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r Requirements.txt
   ```

---

## 🧪 Usage

### LPBF Geometric Monitoring (Current Framework)

#### Running All Tests with Orchestrator

Automatically run all 6 test variants on LPBF layer images and generate consolidated metrics:

```bash
python3 Testing/run_all.py <image_directory> [thickness_threshold] [run_label]
```

**Example**:
```bash
python3 Testing/run_all.py "./data/first layer" 5 lpbf_first_layer
```

**Arguments**:
- `image_directory`: Directory containing layer image files (.png, .jpg, .bmp, etc.)
- `thickness_threshold`: Maximum acceptable wall thickness variance (%, default: 5)
  - Threshold defines anomaly as: thickness deviation > ±thickness_threshold%
  - Lower values = stricter anomaly detection
  - Default 5% suitable for 316L stainless steel thin walls
- `run_label`: Experiment identifier (default: auto-generated timestamp)

**Output**:
- `reports/<run_label>/experiment_summary.txt` — Consolidated comparison table (all 6 models)
- `reports/<run_label>/metrics.json` — Machine-readable metrics
- `reports/<run_label>/*.log` — Individual test logs with detailed feature extraction

**Sample Output Metrics** (experiment_summary.txt):
```
AURA LPBF Geometric Monitoring Experiment Report
================================================================================
Run label: lpbf_first_layer
Image directory: /Users/mcrye/Downloads/AURA/data/first layer
Thickness threshold: ±5%
Started: 2026-04-17 19:20:00 UTC


Running: optical_threshold.py
--------------------------------------------------------------------------------
  Exit code: 0
  Log file: reports/lpbf_first_layer/optical_threshold.log

  Metrics:
    - memory_mb: 274.38
    - detected_anomalies: 39
    - tp: 39
    - fp: 0
    - fn: 0
    - accuracy: 100.00
    - precision: 100.00
    - recall: 100.00


Running: optical_ml.py
--------------------------------------------------------------------------------
  Exit code: 0
  Log file: reports/lpbf_first_layer/optical_ml.log

  Metrics:
    - memory_mb: 140.13
    - actual_anomalies: 8
    - detected_anomalies: 9
    - tp: 8
    - fp: 1
    - fn: 0
    - accuracy: 97.62
    - precision: 88.89
    - recall: 100.00
    - f1: 94.12


Running: timing_threshold.py
--------------------------------------------------------------------------------
  Exit code: 0
  Log file: reports/lpbf_first_layer/timing_threshold.log

  Metrics:
    - memory_mb: 228.50
    - detected_anomalies: 9
    - tp: 9
    - fp: 0
    - fn: 0
    - accuracy: 100.00
    - precision: 100.00
    - recall: 100.00


Running: timing_ml.py
--------------------------------------------------------------------------------
  Exit code: 0
  Log file: reports/lpbf_first_layer/timing_ml.log

  Metrics:
    - memory_mb: 45.04
    - actual_anomalies: 8
    - detected_anomalies: 9
    - tp: 8
    - fp: 1
    - fn: 0
    - accuracy: 97.62
    - precision: 88.89
    - recall: 100.00
    - f1: 94.12


Running: timing_optical_threshold.py
--------------------------------------------------------------------------------
  Exit code: 0
  Log file: reports/lpbf_first_layer/timing_optical_threshold.log

  Metrics:
    - memory_mb: 113.28
    - detected_anomalies: 9
    - tp: 9
    - fp: 0
    - fn: 0
    - accuracy: 100.00
    - precision: 100.00
    - recall: 100.00


Running: timing_optical_ml.py
--------------------------------------------------------------------------------
  Exit code: 0
  Log file: reports/lpbf_first_layer/timing_optical_ml.log

  Metrics:
    - memory_mb: 82.46
    - actual_anomalies: 8
    - detected_anomalies: 9
    - tp: 8
    - fp: 1
    - fn: 0
    - accuracy: 97.62
    - precision: 88.89
    - recall: 100.00
    - f1: 94.12



================================================================================
CONSOLIDATED METRICS COMPARISON
================================================================================

Detection Performance:
  Model                             Accuracy     Precision    Recall      
  -----------------------------------------------------------------------
  optical_ml                               97.62%      88.89%     100.00%
  optical_threshold                       100.00%     100.00%     100.00%
  timing_ml                                97.62%      88.89%     100.00%
  timing_optical_ml                        97.62%      88.89%     100.00%
  timing_optical_threshold                100.00%     100.00%     100.00%
  timing_threshold                        100.00%     100.00%     100.00%


Anomalies Detected:
  Model                             TP       FP       FN       Total   
  ---------------------------------------------------------------
  optical_ml                          8        1        0        9       
  optical_threshold                   39       0        0        39      
  timing_ml                           8        1        0        9       
  timing_optical_ml                   8        1        0        9       
  timing_optical_threshold            9        0        0        9       
  timing_threshold                    9        0        0        9       


Performance Metrics:
  Model                             Memory (MB)    
  --------------------------------------------------
  optical_ml                                 140.13
  optical_threshold                          274.38
  timing_ml                                   45.04
  timing_optical_ml                           82.46
  timing_optical_threshold                   113.28
  timing_threshold                           228.50


Finished: 2026-04-17 19:28:41 UTC
```

#### Running Individual Tests

**Spatial (Optical) Anomaly Detection**:
```bash
python3 Testing/optical_threshold.py ./data/first layer 5
python3 Testing/optical_ml.py ./data/first layer 5
```

**Temporal (Layer-to-Layer) Anomaly Detection**:
```bash
python3 Testing/timing_threshold.py ./data/first layer 5
python3 Testing/timing_ml.py ./data/first layer 5
```

**Hybrid (Spatial + Temporal) Anomaly Detection**:
```bash
python3 Testing/timing_optical_threshold.py ./data/first layer 5
python3 Testing/timing_optical_ml.py ./data/first layer 5
```

**Arguments** (all scripts):
- Arg 1: Directory containing layer images
- Arg 2: Thickness threshold (±%, e.g., 5 = ±5%)

**Outputs** (individual script):
- Excel report (e.g., `optical_threshold_report.xlsx`) with:
  - Per-layer: index, filename, thickness_px, thickness_deviation_%, center_x, center_drift_px, roughness, texture_variance, anomaly_detected (True/False)
  - Summary statistics: mean thickness, std dev, max drift, anomaly count, detection metrics (accuracy, precision, recall)

#### Understanding the Reports

**Geometric Features Explained**:
- **thickness_px** — Wall thickness in pixels (convert to mm by dividing by pixels_per_mm calibration)
- **thickness_deviation_%** — Deviation from baseline thickness as percentage
- **center_drift_px** — Displacement of wall center compared to previous layer (pixel offset)
- **roughness** — Laplacian magnitude (higher = rougher surface)
- **texture_variance** — Pixel intensity variance (higher = more texture/noise)
- **anomaly_detected** — Boolean flag indicating threshold-based anomaly or ML prediction

**Interpreting Detection Results**:
- **Spatial Anomaly** — Indicates wall thickness, drift, or surface roughness outliers in current layer
- **Temporal Anomaly** — Indicates abrupt layer-to-layer change (instability onset)
- **Hybrid Anomaly** — Combined spatial OR temporal criteria triggered (highest recall, lowest false negatives)

**Typical Anomaly Signatures** (per reference [1]):
- Rapid thickness deviation (>±3%) often precedes failure
- Center drift accumulation >0.1mm indicates geometry degradation
- Roughness spikes correlate with surface oxidation or powder adhesion issues
- Temporal models catch early instability; spatial models catch acute defects

---

### Legacy Framework: Direct AES Timing Analysis (Not Updated)

The core framework files (`ML_Detect.py`, `Threshold_Detect.py`, `thermal_adapter.py`) remain available for AES side-channel analysis but are **not actively maintained** for LPBF use cases. Refer to git history or prior documentation for AES-specific usage.

---

## 🔧 LPBF Detection Models Breakdown

### Spatial (Optical-Only) Models
**Purpose**: Detect geometric anomalies from individual layer wall images using the methodology from [1]

**Features Extracted** (per-layer):
- Wall thickness (px), center X position, left/right edge positions
- Surface roughness (Laplacian magnitude), texture variance, contour irregularity
- Thickness deviation (%), center drift (px) vs. rolling baseline

- **`optical_threshold.py`** — Adaptive statistical thresholding on wall geometry
  - Detection: thickness_deviation > ±threshold_% OR roughness > baseline + 2σ
  - Threshold: Auto-computed from data (mean + 2-3σ on each feature)
  - Output: Per-layer anomaly flag + detailed geometric metrics
  - Best for: Quick spatial anomaly identification, deployment with minimal training data

- **`optical_ml.py`** — Random Forest classifier on engineered geometric features
  - Model: RF (100 trees, max_depth=8, random_state=42)
  - Features: thickness_deviation, center_drift, roughness, texture_variance, contour_irregularity
  - Scaling: StandardScaler on feature magnitudes
  - Output: Feature importance scores + anomaly predictions
  - Best for: Learning spatial defect patterns (e.g., powder adhesion zones)

### Temporal (Layer-to-Layer Dynamics) Models
**Purpose**: Detect manufacturing instability from cumulative layer-by-layer evolution

**Features Engineered** (temporal):
- Frame-to-frame thickness changes (%), center drift accumulation (px)
- Rolling statistics: thickness_rolling_std (window=3), center_drift_accumulation
- Velocity metrics: thickness % change per layer, drift acceleration

- **`timing_threshold.py`** — Detects abrupt layer-to-layer deviations
  - Detection: delta_thickness > threshold OR delta_drift_px > baseline + 2σ OR rolling_std > baseline
  - Rationale: "Timing" reinterpreted as temporal evolution; detects instability onset
  - Output: Per-layer anomaly flag + change magnitude metrics
  - Best for: Early warning of process drift (before acute spatial defects appear)

- **`timing_ml.py`** — Random Forest on temporal dynamics features
  - Features: thickness_deviation, center_drift, thickness_rolling_std, thickness_pct_change, drift_accumulation
  - Model: RF (100 trees, max_depth=8)
  - Output: Temporal instability predictions + feature importance
  - Best for: Learning temporal signatures of near-failure conditions

### Hybrid (Spatial + Temporal Fusion) Models
**Purpose**: Combine geometric and temporal signals for robust dual-mode anomaly detection

**Detection Logic**: Trigger anomaly if EITHER spatial OR temporal criterion met

- **`timing_optical_threshold.py`** — Independent thresholds + logical OR fusion
  - Spatial criterion: roughness > threshold OR thickness variance > ±threshold%
  - Temporal criterion: thickness_deviation OR drift_change OR rolling_std
  - Fusion: **spatial_anomaly OR temporal_anomaly → alert**
  - Reports: Both component anomalies + combined flag
  - Best for: Reducing false negatives (high recall); either sensor type can catch defects

- **`timing_optical_ml.py`** — Unified Random Forest on combined 8-feature vector
  - Features: thickness_pct_change, thickness_deviation, center_drift, drift_accumulation, rolling_std, roughness, texture_variance, contour_irregularity
  - Model: RF (100 trees, max_depth=8) trained on hybrid feature space
  - Advantage: ML learns cross-feature interactions (spatial + temporal correlations)
  - Output: Single probabilities with spatial + temporal importance
  - Best for: Highest accuracy; learns subtle multi-modal patterns

### Model Selection Guide

| Scenario | Recommended Model | Reason |
|----------|-------------------|--------|
| **Quick deployment, minimal training** | `optical_threshold.py` | No ML training needed; statistical thresholds from data |
| **Learn spatial defect patterns** | `optical_ml.py` | Identifies texture-based anomalies, powder adhesion zones |
| **Early warning of instability** | `timing_threshold.py` | Detects drift/thickness change before acute defects appear |
| **Detect temporal dynamics** | `timing_ml.py` | Classifies instability signatures (acceleration, jitter) |
| **High recall (catch all defects)** | `timing_optical_threshold.py` | Logical OR: any spatial OR temporal anomaly → alert |
| **Best overall accuracy** | `timing_optical_ml.py` | Learns joint spatial-temporal patterns |
|**Production comparison** | `run_all.py` (all 6 models) | Generates consolidated metrics across all detection modes |

---

## 🧠 Methodology Reference

This framework implements the core in-situ geometric monitoring approach presented in reference [1]:

**Core Technique** (from [1]):
1. Layer-wise optical imaging (20MP monochrome, ~35mm lens, USB 3.0 at 5 fps)
2. Gaussian blur + horizontal intensity profile gradient analysis
3. Wall boundary detection via positive/negative gradient peaks
4. Per-layer: thickness (px), center position, surface roughness (Laplacian)
5. Temporal: frame-to-frame deltas, rolling statistics, cumulative evolution
6. Anomaly detection: multi-criteria thresholding OR ML classification

**Key Findings** (316L stainless steel, per [1]):
- Thickness-dependent instability onset: ~0.6–1.0mm wall thickness
- Critical drift signature: >0.1mm cumulative center drift indicates geometry degradation
- Roughness spikes correlate with surface oxidation and powder adhesion
- Temporal prediction: layer-to-layer changes precede acute spatial defects by 2–4 frames

See the References section below for the full paper citation.

---

## 🛠️ Future Work

- **Thicker wall ranges** — Extend beyond 316L thin walls to medium-thickness (1–5mm) LPBF parts
- **Material expansion** — Adapt to other powders (Inconel, Ti6Al4V, aluminum) with re-calibration
- **Multi-layer correlation** — Develop cross-layer feature dependencies (e.g., Layer N+1 predicted from Layer N trends)
- **Real-time API** — Streaming detection pipeline for closed-loop process control
- **Deep temporal models** — LSTM/GRU for long-range instability prediction (5+ frame lookahead)
- **Hardware acceleration** — GPU-accelerated feature extraction for >10fps processing
- **Thermal integration** — Fuse optical + thermal (IR camera) for subsurface defect detection
- **Porosity correlation** — Post-build CT scan validation against detection predictions
- **Web dashboard** — Real-time monitoring and layer-by-layer anomaly visualization
- **Multi-printer validation** — Test portability across different LPBF systems (Concept Laser, EOS, SLM Solutions)

---

## 📖 References

[1] Killedar, D., Adhami, M., Sun, C., Garcia-Sandoval, M. E., Downey, A. R. J., Fu, Y., Yuan, L., and Parikh, Y., "In-Situ Layer-Wise Geometry Extraction of Thin-Wall 316L Structures Fabricated by Laser Powder Bed Fusion," *Proceedings of the ASME 2025 Annual Review of Progress in Quantitative Nondestructive Evaluation QNDE2026*, paper QNDE2026-191357, September 13–16, 2026, Albuquerque, NM.

```bibtex
@inproceedings{Killedar2026,
  author={Killedar, Digambar and Adhami, Mumin and Sun, Can and Garcia-Sandoval, Mateo E. and Downey, Austin R. J. and Fu, Yanzhou and Yuan, Lang and Parikh, Yash},
  title={In-Situ Layer-Wise Geometry Extraction of Thin-Wall 316L Structures Fabricated by Laser Powder Bed Fusion},
  booktitle={Proceedings of the ASME 2025 Annual Review of Progress in Quantitative Nondestructive Evaluation QNDE2026},
  pages={QNDE2026-191357},
  address={Albuquerque, NM},
  month={September 13--16},
  year={2026}
}
```

---

## 🤝 Acknowledgments

This work was supported under the **McNair Junior Fellowship** and **Magellan Scholar Program** at the University of South Carolina.

Special thanks to **Rye Stahle-Smith** for hardware testing and experimental support on LPBF systems.

Laser Powder Bed Fusion experiments performed on **[LPBF System Model]** at the USC Advanced Manufacturing Lab.

---

## 🎓 Credits

**Framework Adaptation for LPBF**: **Nishant Chinnasami** (2026)  
**Original AES Framework**: **Nishant Chinnasami**  
**Advisor**: **Dr. Rasha Karakchi**  
**Institution**: University of South Carolina  
**Lab**: Advanced Manufacturing & Materials Characterization Lab
