# Industrial Acoustic Anomaly Detection

This repository contains the course project comparing unsupervised and supervised deep learning paradigms on the MIMII industrial fan dataset [1].

## 🧠 Project Overview
The study compares two convolutional architectures sharing the same encoder capacity under physical and environmental domain shifts:
1. **Unsupervised CAE:** Trained strictly on normal operating sounds to reconstruct Log-Mel spectrograms. Uses the 90th percentile of frame-wise MSE for anomaly scoring.
2. **Supervised CNN:** Trained from scratch using labels to output a failure probability (BCE Loss).

Key findings demonstrate that while the supervised CNN achieves a perfect baseline (1.00 AUC at 6dB), it suffers from catastrophic class inversion (0.21 AUC) on an unseen physical unit (ID 06) due to **Shortcut Learning**. The unsupervised CAE, although degraded under noise (0.71 AUC at 0dB), remains logically consistent (0.48 AUC) on unseen domains.

## 📁 Project Structure
```text
├── docs/                 # Documentation and written report
│   └── paper/            # Typst source and compiled PDF report
├── models/               # Trained weights (.pth) (Ignored by Git)
├── results/              # Visualization assets and slide figures
└── src/                  # Modular Python source code
    ├── core/             # Dataset loaders and CAE model definition
    ├── supervised/       # Training and evaluation for the CNN classifier
    ├── unsupervised/     # Training and evaluation for the CAE autoencoder
    └── eval/             # Experimental validation and zero-shot tests
```

## 🛠️ Usage

### 1. Requirements
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running Evaluation Tests
You can run each of the three experimental tests inside `src/eval/` to verify the findings:
```bash
# Generalization test on unseen domain (ID 06)
python3 src/eval/cross_id_test_id06.py

# Noise resilience test at 0dB SNR
python3 src/eval/test_0db_comparison.py

# Multi-scale spectral loss evaluation (Exp 03)
python3 src/eval/test_spectral_id06.py
```

## 📖 References
- [1] H. Purohit et al., "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection," DCASE, 2019.
- [2] R. Tanabe et al., "MIMII DUE: Sound dataset with domain shifts," WASPAA, 2021.
