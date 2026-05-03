# Project Specification: Acoustic Anomaly Detection for Industrial Fans

## 1. Project Overview
This project aims to develop a non-invasive predictive maintenance system based on acoustic signal analysis. Using the **MIMII Dataset**, we implement a **Convolutional Autoencoder (CAE)** to learn the latent representation of normal industrial fan operations. Anomaly detection is based on the reconstruction error (MSE), identifying failures without requiring prior breakage data.

## 2. Technical Objectives
*   **Primary Goal**: Achieve an Area Under the Curve (AUC) of 0.75 or higher in the Fan category.
*   **Metric Strategy**: Use **p-AUC (p=0.1)** as the primary industrial KPI to ensure a low False Positive Rate (FPR).
*   **Scalability**: Implement a "Lazy Loading" pipeline to handle multi-machine data efficiently.

## 3. Dataset Description (MIMII)
*   **Category**: Industrial Fans.
*   **Machine IDs**: 00, 02, 04, 06.
*   **Signal-to-Noise Ratio (SNR)**: 6dB (High noise), 0dB, and -6dB (Extreme noise).
*   **Audio Format**: 16kHz Mono, PCM (.wav).

## 4. Proposed Architecture (CAE)
The system utilizes a 4-layer Convolutional Autoencoder:
*   **Encoder**: 2D Convolutions with stride-2 for spatial reduction, extracting spectro-temporal features.
*   **Bottleneck**: A compressed latent space with Dropout regularization (0.2) to prevent noise memorization.
*   **Decoder**: Transposed Convolutions to reconstruct the original Log-Mel spectrogram.
*   **Output**: Linear activation (to support Z-score range) and deterministic cropping to match input dimensions (128x313).

## 5. Preprocessing Pipeline
1.  **STFT**: Hamming window (1024), 50% overlap (512 hop).
2.  **Mel Scaling**: 128 Mel bands.
3.  **Log-Transformation**: Power-to-dB conversion.
4.  **Frequency-wise Z-Score**: Independent normalization for each frequency band using global training statistics (Mean/Std).

## 6. Implementation Strategy
*   **Phase 1**: Environment setup and data verification.
*   **Phase 2**: Development of the `MIMIIDataset` class with frequency-wise normalization.
*   **Phase 3**: CAE Model design and training on "Normal" data only (Unsupervised).
*   **Phase 4**: Comparative evaluation using AUC and p-AUC metrics.
*   **Phase 5**: Domain generalization stress tests (Cross-ID and Cross-SNR).
