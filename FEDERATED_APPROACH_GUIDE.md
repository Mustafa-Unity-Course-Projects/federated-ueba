# F-UEBA: Federated User and Entity Behavior Analytics

This document outlines the technical architecture and advanced anomaly detection logic implemented in this Federated Learning (FL) framework. The goal is communication-efficient, privacy-preserving insider threat detection using a **Bidirectional LSTM-Autoencoder**.

---

## 1. System Architecture
The system utilizes the **Flower (flwr)** framework to coordinate learning across multiple decentralized clients.

*   **Server**: Aggregates local model weights using the `FedAvg` strategy and manages global checkpoints.
*   **Clients**: Train a local LSTM-Autoencoder on their respective user partitions. Each client maintains its own behavioral baseline to handle **Non-IID (Independent and Identically Distributed)** behavioral data.

---

## 2. Advanced Model Architecture
The core model is a **Bidirectional LSTM-Autoencoder** (128-dimensional hidden space) designed for robustness:

*   **Bidirectional Encoder**: Captures temporal dependencies from both past and future context within a window.
*   **Deeper Bottleneck**: Uses multiple `Linear` layers with `LayerNorm` and `ReLU` activation to learn complex non-linear behavioral representations.
*   **Denoising Mechanism**: Random Gaussian noise is added during training to prevent the model from memorizing specific data points, forcing it to learn the underlying distribution of "Normal" behavior.

---

## 3. The Data Pipeline
To ensure high precision in imbalanced environments, the following preprocessing steps are applied:

1.  **Log-Transformation (`log1p`)**: Stabilizes variance across diverse user activities and compresses the range of high-frequency events (e.g., HTTP counts).
2.  **Local Scaling (`StandardScaler`)**: Each client fits a scaler **only on the normal behavior** of its assigned users. This prevents "global data leakage" and ensures the model learns what is specifically normal for that local sub-population.
3.  **Train/Validation Split (Anti-Overfitting)**: Each client reserves **20% of its normal data** as a held-out validation set. This allows for real-time monitoring of the "Generalization Gap" during training.

---

## 4. Advanced Anomaly Detection Logic (Inference)
The system goes beyond simple reconstruction error by using a **Multi-Vector Deviation** strategy:

### A. Local Behavioral Calibration
Instead of a global Z-score, each client calculates a **Local Baseline** (`mean` and `std`) of **Squared Reconstruction Errors** during the training phase. This allows the system to be sensitive to subtle changes relative to a specific user's peer group.

### B. Top-K Feature Focus
Insider threats often manifest in a subset of activities (e.g., only USB and Email). The system identifies the **Top 5 most deviating features** in every window, ensuring that a massive spike in one area isn't "diluted" by 40 other normal features.

### C. Diversity Factor
This metric boosts the anomaly score if **multiple independent features** deviate simultaneously (e.g., Email, USB, and File access at once).

### D. Persistence Window
To reduce false positives from one-off "noisy" days, the system averages the **3 worst windows** for each user. This ensures that only sustained behavioral shifts are flagged as high-priority threats.

---

## 5. Model Selection & Evaluation
Because insider threats are rare (highly imbalanced), we use metrics that are "Imbalance-Aware":

*   **Precision-Recall AUC (PR-AUC)**: The primary metric for model selection.
*   **Balanced Accuracy**: The average of recall for both the "Insider" and "Normal" classes. This prevents the "Accuracy Paradox" where 96% accuracy can mean zero detection.
*   **Confusion Matrix**: Provides explicit counts of True Positives (caught), False Negatives (missed), and False Positives (noise).
*   **Multi-Round Search**: Automatically scans every 10th round to find the "Winner Round" with the best PR-AUC performance.

---

## 6. Execution Workflow

1.  **Cleanup**: Every run automatically clears previous models and logs.
2.  **Training**: Clients train locally while monitoring both **Train Loss** and **Val Loss** to verify learning progress without overfitting.
3.  **Search & Detection**: `federated_insider_detection.py` scans the checkpoints, applies the advanced logic, and generates a comprehensive performance report in the `federated_evaluation_reports` folder.

---
**Goal**: Achieving >90% Precision for insider threat detection while minimizing communication overhead through federated aggregation.
