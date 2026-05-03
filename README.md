# Industrial Fan Anomaly Detection (CAE)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-precision predictive maintenance system based on **Unsupervised Deep Learning**. This project implements a **Convolutional Autoencoder (CAE)** to detect mechanical failures in industrial fans using acoustic signals from the **MIMII Dataset**.

## 🚀 Key Achievements
*   **State-of-the-Art Performance**: Achieved **0.9880 ROC-AUC** and **0.9469 p-AUC** on Machine ID 02.
*   **Industrial Robustness**: Implements frequency-wise Z-score normalization to handle high background noise (6dB SNR).
*   **Optimized Pipeline**: On-the-fly spectrogram processing with PyTorch `DataLoader` to ensure low memory footprint.

---

## 🧠 System Architecture
The system uses a non-invasive approach to model "Normal" sound signatures and identify deviations as potential failures.

1.  **Preprocessing**: Converts raw 1D audio into 2D **Log-Mel Spectrograms**.
2.  **Normalization**: Frequency-wise Z-score scaling (Independent Mean/Std for each Mel band).
3.  **Model**: A 4-layer 2D Convolutional Autoencoder with **Dropout** for regularization.
4.  **Anomaly Scoring**: Based on the **90th percentile of Reconstruction MSE** across temporal frames.

---

## 📁 Project Structure
```text
.
├── data/               # Raw MIMII audio files (Not tracked by Git)
├── docs/               # Professional technical documentation
│   ├── technical/      # Project Specification & Development Log
│   └── paper/          # LaTeX source for the final report
├── models/             # Trained weights (.pth) and metadata
├── results/            # Visualization of loss and metrics
├── src/                # Modular Python source code
│   ├── dataset.py      # Audio processing & Lazy Loading
│   ├── model_ae.py     # CAE Architecture
│   ├── train.py        # Unsupervised training loop
│   └── eval.py         # AUC/p-AUC evaluation & visualization
├── run_pipeline.sh     # Orchestrator script
└── requirements.txt    # Environment dependencies
```

---

## 🛠️ Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ and pip installed.
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Place your MIMII dataset files in the following structure:
`data/6_dB_fan/fan/id_XX/[normal|abnormal]/*.wav`

### 3. Run the Full Pipeline
The orchestrator script handles training and evaluation in one command:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

---

## 📊 Results (Machine ID 02)
The model demonstrates exceptional class separation by focusing on frequency-specific deviations.

| Metric | Score | Industrial Impact |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.9880** | Extremely high discriminative power. |
| **p-AUC (0.1)** | **0.9469** | Very low false alarm rate, critical for factory uptime. |

> **Note**: For a detailed history of all experiments and the root cause analysis of initial failures, see the [Development Log](docs/technical/DEVELOPMENT_LOG.md).

---

## 📖 Documentation
Detailed technical descriptions are available in the `docs/` folder:
*   [Project Specification](docs/technical/PROJECT_SPECIFICATION.md): Mathematical foundations and architecture.
*   [Development Log](docs/technical/DEVELOPMENT_LOG.md): Iterations, hypotheses, and analysis.

## 📜 References
*   **MIMII Dataset**: Purohit et al. (2019). Sound dataset for malfunctioning industrial machine investigation and inspection.
*   **DCASE Challenge**: Standard benchmarks for Anomalous Sound Detection.
