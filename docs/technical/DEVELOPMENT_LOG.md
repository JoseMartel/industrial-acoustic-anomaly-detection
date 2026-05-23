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

## Iteration 4: Multi-ID Training (IDs 00, 02, 04)
*   **Status**: Completed
*   **Configuration**: Combined training with normal sounds from IDs 00, 02, and 04. Evaluation using combined testing sets of the same IDs.
*   **Hypothesis**: The model can generalize and identify anomalies across different physical machines of the same type (Fan).
*   **Results**: **ROC-AUC: 0.8154**, **p-AUC: 0.7060**
*   **Analysis**: 
    *   **Confirmation**: As seen in logs, exactly 20% of normal files and all abnormal files from IDs 00, 02, and 04 were used for evaluation.
    *   **Performance Drop**: There is a significant drop (~17%) compared to the specialized ID 02 model. This suggests that "normality" varies between machines, and a single global normalization (Mean/Std across all IDs) might be blurring the distinctive features of each machine.
    *   **Challenge**: The model now faces a more complex distribution of "normal" sound, making the reconstruction task harder for the bottleneck.

## Planned Iterations & Research Roadmap

### Iteration 5: SNR Sensitivity & Noise Verification
*   **Goal**: Confirm if background noise (SNR) is the primary factor degrading multi-ID performance.
*   **Method**: Evaluate the current Iteration 4 model across different SNR levels available in the MIMII dataset (6dB, 0dB, -6dB).
*   **Expected Metric**: A significant AUC drop at lower SNR would confirm the need for noise-robust features.

### Iteration 6: Machine-Agnostic Generalization (DANN)
*   **Goal**: Improve multi-ID AUC (currently 0.81) and enable support for unseen Machine IDs.
*   **Technique**: **Domain Adversarial Neural Networks (DANN)**.
*   **Implementation**: 
    *   Add a Gradient Reversal Layer (GRL) connected to the bottleneck.
    *   Train a secondary classifier to predict Machine ID, forcing the Encoder to learn ID-invariant features.

### Iteration 7: Invariant Feature Engineering
*   **Goal**: Reduce sensitivity to absolute energy levels and machine-specific "bias".
*   **Methods**:
    *   **Instance Normalization**: Replace BatchNorm with InstanceNorm to focus on spectrogram "texture" rather than global energy.
    *   **Temporal Dynamics**: Incorporate Delta and Delta-Delta coefficients to the input features to highlight temporal changes over static background noise.

### Long-term: Zero-Shot / ID-10 Support
*   **Strategy**: Implementation of a 5-second "Reference Calibration" step. Instead of retraining, use a small window of normal operation to dynamically adjust the anomaly threshold for new units.
