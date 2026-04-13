# **AURA: AES Utility for Real-time Anomaly Detection**
### *Framework for Side-Channel Analysis and Malicious Block Detection*

---

## 📌 Project Overview

**AURA** is a lightweight, modular framework for detecting anomalies in AES-128 encryption operations through multi-modal side-channel analysis. The framework is designed to identify malicious encryption blocks injected into embedded systems by analyzing timing, thermal, and optical side-channels. It supports both direct AES execution timing analysis and image-based signal extraction from optical camera datasets, making it suitable for real-time anomaly detection in embedded systems, SoCs, and FPGAs.

The framework implements three primary detection strategies:
- **Threshold-based detection** — Statistical anomaly identification using adaptive thresholds
- **ML-based detection** — Machine learning classifiers (Random Forest) for pattern recognition
- **Hybrid detection** — Fusion of both approaches for improved accuracy

---

## ⚙️ Features

- 🔐 **AES-128 Encryption** — ECB mode via PyCryptodome with timed execution analysis
- 📊 **Multi-Modal Feature Extraction**:
  - Timing features (execution time, deltas, rolling statistics)
  - Optical features (mean pixel intensity from images via OpenCV)
  - Fused multi-modal feature vectors for ensemble methods
- 🎯 **Three Detection Modes**:
  - **Threshold Detection** — Adaptive statistical thresholding (mean + 3-sigma)
  - **ML Detection** — Random Forest classifier with temporal feature engineering
  - **Hybrid Detection** — Logical OR combination of both methods
- ⚙️ **Multi-Core Parallel Processing** — Leverages multiprocessing for high throughput
- 📷 **Image-Based Dataset Support** — Frame-by-frame optical signal extraction from directories
- 📈 **Automated Reporting** — Excel (.xlsx) export with detailed metrics and statistics
- 🏃 **Real-Time Ready** — Low-latency inference designed for embedded deployment

---

## 📂 Repository Structure

```
AURA/
├── README.md                    # Project documentation
├── Requirements.txt             # Python dependencies
├── ML_Detect.py                 # Core ML-based anomaly detector (MAIN FRAMEWORK)
├── Threshold_Detect.py          # Core threshold-based detector (MAIN FRAMEWORK)
├── thermal_adapter.py           # Video/image adapter for side-channel signals (MAIN FRAMEWORK)
│
└── Testing/                     # Comprehensive testing and ablation studies
    ├── _utils.py                # Shared utilities (image loading, feature engineering)
    ├── run_all.py               # Orchestrator for running all experiments
    │
    ├── timing_threshold.py      # Test: Timing-only threshold detection
    ├── timing_ml.py             # Test: Timing-only ML detection
    │
    ├── optical_threshold.py     # Test: Optical-only threshold detection
    ├── optical_ml.py            # Test: Optical-only ML detection
    │
    ├── timing_optical_threshold.py  # Test: Fused timing+optical threshold
    └── timing_optical_ml.py         # Test: Fused timing+optical ML
```

### Core Framework Files (Root Directory)

The three primary files form the **main detection framework**:

| File | Purpose | Workflow |
|------|---------|----------|
| **ML_Detect.py** | ML-based anomaly classification | AES blocks → timing extraction → RF classifier → predictions |
| **Threshold_Detect.py** | Statistical threshold-based detection | AES blocks → timing measurement → adaptive threshold → binary detection |
| **thermal_adapter.py** | Video/image to anomaly detector bridge | Video/image input → frame signal extraction → call ML_Detect or Threshold_Detect → report |

These are designed for **direct AES timing analysis** on live encryption operations or arbitrary signal-based inputs.

### Testing Suite (Testing/ Directory)

Six specialized test scripts for **modular sensor evaluation** and **ablation studies** using image datasets:

| Category | Script | Input | Method |
|----------|--------|-------|--------|
| **Timing Only** | `timing_threshold.py` | Simulated AES + synthetic timing | Threshold-based |
| | `timing_ml.py` | Simulated AES + synthetic timing | Random Forest |
| **Optical Only** | `optical_threshold.py` | Real images (mean pixel intensity) | Threshold-based |
| | `optical_ml.py` | Real images (mean pixel intensity) | Random Forest |
| **Timing + Optical (Fused)** | `timing_optical_threshold.py` | Images + AES simulation (dual-channel) | Threshold-based |
| | `timing_optical_ml.py` | Images + AES simulation (dual-channel) | Random Forest |

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.7+**
- **Virtual environment** (recommended: `venv` or `conda`)
- **Dependencies**: `pycryptodome`, `pandas`, `numpy`, `scikit-learn`, `opencv-python`, `openpyxl`, `psutil`

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

### Core Framework: Direct AES Timing Analysis

#### Threshold-Based Detection
Run timing-based anomaly detection on simulated AES encryption:

```bash
python3 Threshold_Detect.py
```

**Prompts** (interactive):
- Number of plaintext blocks to encrypt
- Percentage of malicious blocks to inject (0-100)
- Number of CPU cores to use

**Output**: `aes_anomaly_report.xlsx` — Detailed report with timing measurements, detections, and metrics

#### ML-Based Detection
Run Random Forest classification on AES timing features:

```bash
python3 ML_Detect.py
```

**Same interactive prompts as Threshold_Detect.py**

**Output**: `ml_aes_anomaly_report.xlsx` — ML predictions with feature importance rankings

#### Video/Image Adapter
Process optical camera images or video files:

```bash
python3 thermal_adapter.py <path_to_video_or_image>
```

**Example**:
```bash
python3 thermal_adapter.py "/path/to/optical_camera_footage.mp4"
```

**Prompts**:
- Detection mode: `threshold`, `ml`, or `hybrid`

**Output**: `<filename>_thermal_report.xlsx` — Frame-by-frame anomaly detection results

---

### Testing Suite: Sensor Ablation Studies

Use the Testing suite to evaluate individual sensors or sensor combinations on a directory of optical camera images.

#### Running Individual Tests

**Timing-Only Threshold Detection**:
```bash
python3 Testing/timing_threshold.py /path/to/optical_images 20 4
```
- Arg 1: Directory containing image files
- Arg 2: Anomaly injection rate (%, default: 20)
- Arg 3: Number of worker threads (default: 4)

**Optical-Only ML Detection**:
```bash
python3 Testing/optical_ml.py /path/to/optical_images 20
```

**Timing + Optical Fused Threshold**:
```bash
python3 Testing/timing_optical_threshold.py /path/to/optical_images 20 4
```

#### Running All Tests with Orchestrator

Automatically run all 6 test variants and generate consolidated metrics:

```bash
python3 Testing/run_all.py /path/to/optical_images 20 experiment_001
```

**Arguments**:
- `image_directory`: Directory containing optical camera images (.jpg, .png, .bmp, etc.)
- `anomaly_percent`: Injection rate for anomalies (default: 20)
- `run_label`: Experiment identifier (default: timestamp)

**Output**:
- `reports/experiment_001/experiment_summary.txt` — Consolidated comparison table
- `reports/experiment_001/metrics.json` — Machine-readable metrics
- `reports/experiment_001/*.log` — Individual test logs

**Sample Consolidated Comparison**:
```
Accuracy Comparison:
  Model                            Accuracy     Precision    Recall       F1-Score
  ───────────────────────────────────────────────────────────────────────────
  timing_ml                          95.60%       94.20%       96.50%       95.30%
  timing_optical_ml                  98.10%       97.80%       98.40%       98.10%
  optical_ml                         92.30%       91.50%       93.10%       92.30%
  timing_optical_threshold           85.40%       86.20%       84.60%       85.40%
  timing_threshold                   82.10%       80.90%       83.30%       82.10%
  optical_threshold                  78.50%       77.60%       79.40%       78.50%

Detection Statistics:
  Model                            TP       FP       FN       Total
  ──────────────────────────────────────────────────────────────
  timing_ml                        191      10       9        210
  timing_optical_ml                196      4        4        204
  optical_ml                       186      18       14       218
  ...
```

---

## 🔧 Testing Models Breakdown

### Timing-Only Models
**Purpose**: Evaluate pure timing-side-channel effectiveness without optical data

- **`timing_threshold.py`** — Adaptive statistical thresholding on AES execution timing
  - Features: delta_prev, delta_next, rolling_mean, rolling_std
  - Threshold: mean + 3-sigma
  - Best for: Resource-constrained environments

- **`timing_ml.py`** — Random Forest trained on temporal timing patterns
  - Features: Same as threshold + engineered temporal features
  - Model: RF (100 trees, random_state=42)
  - Output: Feature importance rankings

### Optical-Only Models
**Purpose**: Evaluate optical side-channel (pixel intensity) effectiveness on real images

- **`optical_threshold.py`** — Threshold detection on mean image brightness
  - Features: Extracted from OpenCV grayscale images
  - Process: frame_mean > threshold → anomaly
  - Simulates optical glitch signatures

- **`optical_ml.py`** — Random Forest on optical spatial/temporal patterns
  - Features: delta_prev, delta_next, rolling statistics of pixel intensity
  - Handles: Real image variations and noise

### Timing + Optical Fused Models
**Purpose**: Evaluate multi-modal fusion for improved detection

- **`timing_optical_threshold.py`** — Logical OR of independent thresholds
  - Detection: (timing > threshold_t) OR (optical > threshold_o)
  - Advantage: Either sensor can trigger alert
  - Lower false negatives

- **`timing_optical_ml.py`** — Single Random Forest on combined feature set
  - Features: All timing + all optical features in unified vector
  - Advantage: ML learns cross-modal interactions
  - Typically highest accuracy

---

## 🛠️ Future Work

- Expand dataset to include more sensor modalities (event cameras, EMI)
- Explore deeper neural networks (LSTM, Transformer) for temporal modeling
- Implement quantization for edge deployment
- Develop real-time streaming API
- Add support for hardware acceleration (GPU, FPGA)
- Create web dashboard for monitoring
- Expand to other encryption algorithms (RSA, ECC)

---

## 🤝 Acknowledgments

This work was supported under the **McNair Junior Fellowship** and **Magellan Scholar Program** at the University of South Carolina.

Special thanks to **Rye Stahle-Smith** for hardware testing and experimental support.

---

## 🎓 Credits

Developed by **Nishant Chinnasami**  
Advisor: **Dr. Rasha Karakchi**  
University of South Carolina
