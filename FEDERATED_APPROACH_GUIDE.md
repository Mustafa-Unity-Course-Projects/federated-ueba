# F-UEBA: Federated User and Entity Behavior Analytics

This document outlines the technical architecture and advanced anomaly detection logic implemented in this Federated Learning (FL) framework. The goal is communication-efficient, privacy-preserving insider threat detection using a **Bidirectional LSTM-Autoencoder**.

---

## 1. System Architecture
The system utilizes the **Flower (flwr)** framework to coordinate learning across multiple decentralized clients.

*   **Server**: Aggregates local model weights using the `FedAvg` strategy and manages global checkpoints.
*   **Clients**: Train a local LSTM-Autoencoder on their respective user partitions. Each client maintains its own behavioral baseline to handle **Non-IID (Independent and Identically Distributed)** behavioral data.

---

## 2. The Data Pipeline
To ensure high precision in imbalanced environments, the following preprocessing steps are applied:

1.  **Log-Transformation (`log1p`)**: Stabilizes variance across diverse user activities and compresses the range of high-frequency events (e.g., HTTP counts).
2.  **Local Scaling (`StandardScaler`)**: Each client fits a scaler **only on the normal behavior** of its assigned users. This prevents "global data leakage" and ensures the model learns what is specifically normal for that local sub-population.
3.  **Sequence Building**: Sliding windows (default: 14 days) are used to capture temporal dependencies in user behavior.

---

## 3. Advanced Anomaly Detection Logic (Inference)
The system goes beyond simple reconstruction error by using a **Multi-Vector Deviation** strategy:

### A. Local Behavioral Calibration
Instead of a global Z-score, each client calculates a **Local Baseline** (`mean` and `std`) of reconstruction errors during the training phase. This allows the system to be sensitive to subtle changes relative to a specific user's peer group.

### B. Top-K Feature Focus
Insider threats often manifest in a subset of activities (e.g., only USB and Email). The system identifies the **Top 5 most deviating features** in every window, ensuring that a massive spike in one area isn't "diluted" by 40 other normal features.

### C. Diversity Factor
This metric boosts the anomaly score if **multiple independent features** deviate simultaneously. If a user is acting strangely in Email, USB, and File access at once, the `Diversity Factor` exponentially increases their threat score.

### D. Persistence Window
To reduce false positives from one-off "noisy" days, the system averages the **3 worst windows** for each user. This ensures that only sustained behavioral shifts are flagged as high-priority threats.

---

## 4. Model Selection & Evaluation
Because insider threats are rare (imbalanced data), the system ignores Accuracy and ROC-AUC in favor of:

*   **Precision-Recall AUC (PR-AUC)**: The primary metric for model selection. It focuses strictly on how well the model identifies the "positive" (insider) class without flagging thousands of normal users.
*   **Max F1-Score**: Used to identify the optimal detection threshold for each specific round of training.
*   **Multi-Round Search**: The detection script automatically scans every 10th round to find the "Winner Round" that provides the best PR-AUC performance.

---

## 5. Execution Workflow

1.  **Cleanup**: Every `flwr run` automatically clears previous models and logs to ensure experimental integrity.
2.  **Training**: Clients train locally and save their **Local Calibration Stats** (`error_stats_client_{i}.pkl`).
3.  **Search & Detection**: `federated_insider_detection.py` scans the checkpoints, applies the advanced multi-vector logic, and generates a comprehensive performance report in the `federated_evaluation_reports` folder.

---
**Goal**: Achieving >90% Precision for insider threat detection while minimizing communication overhead through federated aggregation.
