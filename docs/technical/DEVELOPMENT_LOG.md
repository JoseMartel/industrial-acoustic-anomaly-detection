# Development Log: Experimentation & Iterations

## Iteration 1: Baseline Architecture with Local Normalization
*   **Status**: Completed (Failed)
*   **Configuration**: CAE Simple, Log-Mel Spectrogram, Local Min-Max Scaling.
*   **Hypothesis**: A basic autoencoder will distinguish between normal and abnormal sounds.
*   **Results**: **ROC-AUC: 0.47**
*   **Root Cause Analysis**: Local normalization (per file) erased absolute volume differences. Anomalies that were louder than normal sounds were scaled down, becoming "invisible" to the model.

## Iteration 2: Global Normalization (Partial Subset)
*   **Status**: Completed (Marginal Improvement)
*   **Configuration**: Global Min-Max (subset of 100 samples).
*   **Hypothesis**: Preserving relative volume across files will improve class separation.
*   **Results**: **ROC-AUC: 0.56**
*   **Analysis**: Significant improvement, but the model still struggled with background noise (6dB SNR) and the reconstruction error was diluted by averaging over 10 seconds.

## Iteration 3: Frequency-wise Z-Score & Percentile 90 Scoring
*   **Status**: Completed (Target Achieved)
*   **Configuration**: 
    *   **Z-Score by Frequency**: Independent Mean/Std for each of the 128 Mel bands.
    *   **Regularization**: Dropout (0.2) in the bottleneck.
    *   **Scoring**: 90th Percentile of MSE across temporal frames (instead of Mean).
*   **Hypothesis**: Frequency-wise scaling will highlight high-frequency anomalies, and Percentile scoring will detect transient faults while ignoring random noise.
*   **Results (ID 02)**: **ROC-AUC: 0.9880**, **p-AUC: 0.9469**
*   **Conclusion**: The system is highly robust for Machine ID 02. The frequency-wise normalization proved to be the most impactful technical change.

## Planned Iterations
*   **Iteration 4 (Next)**: Multi-ID Training (Generalization). Training on IDs 00, 02, 04 and testing on ID 06.
*   **Iteration 5**: Comparative Supervised Learning. CNN Classifier vs. Autoencoder to demonstrate domain adaptation failure in supervised models.
