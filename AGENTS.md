Project Context: F-UEBA Federated Learning

Project Overview:
Goal: Communication-Efficient Anomaly Detection in Federated Learning-Based User and Entity Behavior Analytics.
Method: Flower (flwr) framework.
Model: LSTM-Autoencoder.
Performance Target: 90% Precision for insider threat detection.
Environment: WSL2 (Ubuntu) on Windows, NVIDIA GPU/CUDA enabled.

Technical Stack:
Language: Python 3.
Libraries: pandas, torch, pickle, sklearn, flwr, ray.
Communication: Flower SuperLink on 127.0.0.1:39093.

Data Specifications:
Source File: weekr4.2-percentile30.pkl
Feature Count: Dynamic (Detected by Scaler).
Loading Method: Binary stream via pickle.load().
Data Partitioning: Dynamic (np.array_split based on num_clients).
Scaler Storage: ./scaler_data/scaler_client_{i}.pkl
Logic: Local scaling handles Non-IID behavioral baselines and prevents global data leakage.

Architectural Logic:

Training: Aggregates local LSTM weights to a central server via Flower.

Global Model: parameters_round_20.pkl (or highest round) stored in ./model_pickle.

Inference Pairing: Input Data -> Local Client Scaler -> Global Model Weights -> Reconstruction Error.

Anomaly Detection: Z-Score based on Reconstruction Mean Squared Error (MSE).
Threshold: Dynamic Z-Score (e.g., 3.5) with sensitivity analysis.

---
Agent Instructions:
-   Do not modify files unnecessarily.
-   Always check project context from this file before performing edits.
-   Maintain consistency with the project's technical stack and architectural logic.
