# AURA (AES Utility for Real-time Anomaly Detection)

AURA is a lightweight real-time anomaly detection framework developed for AES-128 encryption timing analysis on embedded systems such as SoCs and FPGAs.

#### The framework supports:

- Statistical threshold-based anomaly detection
- Machine learning–based anomaly detection
- Hybrid threshold + ML detection
- Cross-modality dataset benchmarking (thermal, optical, event camera)

---

### 🚀 Features

- AES-128 Encryption (PyCryptodome, ECB mode)
- Adaptive Threshold-Based Detection
- ML-Based Detection (Random Forest classifier)
- Hybrid Detection Mode (Threshold + ML)
- Multi-Core Parallel Execution
- Cross-Modality Video Dataset Support
- Automatic Excel Report Generation
- Embedded-System Ready Design

---

### 🧠 System Overview

AURA supports two primary workflows.

1. AES Timing Anomaly Detection

```
AES Execution
↓
Timing Feature Extraction
↓
Threshold / ML / Hybrid Detection
↓
Anomaly Report
```

2. Video-Based Dataset Benchmarking

```
Video Input (Thermal / Optical / Event)
↓
Frame-to-frame statistical signal extraction
↓
Threshold / ML / Hybrid Detection
↓
Structured anomaly report
```

#### The core detector remains unchanged. The video adapter converts frame dynamics into a statistical signal compatible with the existing detection pipeline.

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
   git clone https://github.com/Karakchi-Research/AURA.git
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

🧪 Usage

### AES Detection Experiments

#### Threshold-Based AES Detection:

```python
python3 Threshold_Detect.py
```

#### ML-Based AES Detection:

```python
python3 ML_Detect.py
```

#### Results are saved as `aes_anomaly_report.xlsx`

---

### 📹 Video Dataset Benchmarking

#### Run anomaly detection on thermal, optical, or event camera videos:

```python
python3 thermal_adapter.py <path_to_video_file>
```

**Example**:

```python
python3 thermal_adapter.py "/path/to/Laser Welds.mp4"
```

#### The program will prompt you to select the detection mode:

- threshold
- ml
- hybrid

---

### ⚙️ Mode Descriptions:

- **threshold** → Statistical timing anomaly detection
- **ml** → Random Forest ML anomaly detection
- **hybrid** → Combines threshold and ML detection

#### The script will:

- Process video frame-by-frame
- Extract frame-change statistical signals
- Run the selected anomaly detection mode
- Generate an Excel report

#### Output will be saved as `<video_name>_thermal_report.xlsx`

---

## 📂 Repository Structure

```
AURA/
├── README.md                    # Project documentation
├── Requirements.txt             # Python dependencies
├── ML_Detect.py                 # ES-based ML detector (kept for compatibility)
├── Threshold_Detect.py          # AES-based threshold detector (kept for compatibility)
├── thermal_adapter.py           # Signal adapter (kept for compatibility)
└── Testing/                     # Test suite
```

---

## 🎓 Credits

**Developed by**: **Nishant Chinnasami**  
**Advisor**: **Dr. Rasha Karakchi**  
**Institution**: University of South Carolina

---

## 🤝 Acknowledgments

This work was supported under the **McNair Junior Fellowship** and **Magellan Scholar Program** at the University of South Carolina.

Special thanks to **Rye Stahle-Smith** for hardware testing and experimental support.

---
