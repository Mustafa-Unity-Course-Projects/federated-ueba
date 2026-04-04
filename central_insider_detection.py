import os
import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import federated_ueba.task as task
from config_manager import config

# --- CONFIGURATION ---
CENTRALIZED_MODEL_PATH = "centralized_model.pth"
CENTRALIZED_SCALER_PATH = "centralized_scaler.pkl"
ERROR_STATS_PATH = "centralized_error_stats.pkl"
DATA_PATH = config.get("data", "processed_data_path")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCAN_STRIDE = 1 
SELECTED_FEATURES = config.get("data", "selected_features") or None
HIDDEN_DIM = config.get("model", "hidden_dim") or 64
TOP_K_FEATURES = 5
PERSISTENCE_WINDOW = 3 # Average the top 3 worst windows
DIVERSITY_THRESHOLD = 2.0 # Minimum Z-score to consider a feature "deviating"

def run_zscore_scan():
    print(f"🚀 Starting Multi-Vector Bidirectional Scan on {DEVICE}...")

    # 1. Load Data
    if DATA_PATH.endswith(".pkl"):
        df = pd.read_pickle(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH, low_memory=False)
    
    time_col = 'day' if 'day' in df.columns else ('week' if 'week' in df.columns else None)
    df = df.sort_values(['user', time_col])

    # 2. Load Stats, Scaler & Model
    with open(CENTRALIZED_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    
    expected_features = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else SELECTED_FEATURES
    model = task.LSTMAutoencoder(input_dim=len(expected_features), hidden_dim=HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load(CENTRALIZED_MODEL_PATH, map_location=DEVICE))
    model.eval()

    with open(ERROR_STATS_PATH, "rb") as f:
        stats = pickle.load(f)
        mean_per_feature = stats["mean_per_feature"]
        std_per_feature = stats["std_per_feature"]
        topk_metric_mean = stats["topk_metric_mean"]
        topk_metric_std = stats["topk_metric_std"]

    # 3. Scanning Loop
    results = []
    unique_users = df['user'].unique()

    with torch.no_grad():
        for user in unique_users:
            user_df = df[df['user'] == user]
            u_features = user_df.reindex(columns=expected_features, fill_value=0)
            u_features = u_features.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
            u_features = np.log1p(u_features.clip(lower=0))
            u_scaled = scaler.transform(u_features)
            user_tensor = torch.tensor(u_scaled, dtype=torch.float32).to(DEVICE)

            user_window_metrics = []
            if len(user_tensor) >= task.WINDOW_SIZE:
                for i in range(0, len(user_tensor) - task.WINDOW_SIZE + 1, SCAN_STRIDE):
                    window = user_tensor[i : i + task.WINDOW_SIZE].unsqueeze(0)
                    reconstruction = model(window)
                    
                    # Calculate per-feature deviation
                    per_feature_abs_error = torch.mean(torch.abs(reconstruction - window), dim=1).squeeze(0).cpu().numpy()
                    feat_z = (per_feature_abs_error - mean_per_feature) / (std_per_feature + 1e-6)
                    
                    # TOP-K LOGIC with DIVERSITY FACTOR
                    pos_feat_z = np.maximum(feat_z, 0)
                    top_k_z = np.sort(pos_feat_z)[-TOP_K_FEATURES:]
                    
                    # --- NEW: COOPERATIVE DEVIATION BOOST ---
                    # How many features are significantly deviating (> 2 std devs)?
                    num_anomalous_features = np.sum(pos_feat_z > DIVERSITY_THRESHOLD)
                    diversity_factor = 1.0 + (num_anomalous_features / len(expected_features))
                    
                    window_metric = np.mean(top_k_z) * diversity_factor
                    user_window_metrics.append(window_metric)

            has_insider_activity = (user_df['insider'] != 0).any()
            
            if user_window_metrics:
                user_window_metrics = np.sort(np.array(user_window_metrics))
                top_metrics_avg = np.mean(user_window_metrics[-min(len(user_window_metrics), PERSISTENCE_WINDOW):])
                final_z_score = (top_metrics_avg - topk_metric_mean) / (topk_metric_std + 1e-6)
            else:
                final_z_score = 0.0

            results.append({
                "user": user,
                "max_z_score": final_z_score,
                "is_actual_insider": 1.0 if has_insider_activity else 0.0
            })

    results_df = pd.DataFrame(results).sort_values(by="max_z_score", ascending=False)
    results_df.to_csv("centralized_insider_results.csv", index=False)
    return results_df

def generate_report(df, thresholds=[5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0], output_dir="centralized_evaluation_reports"):
    os.makedirs(output_dir, exist_ok=True)
    metrics_data = []

    for threshold in thresholds:
        df['predicted'] = (df['max_z_score'] >= threshold).astype(float)
        y_true, y_pred = df['is_actual_insider'], df['predicted']
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        metrics_data.append({'Threshold': threshold, 'Precision': precision, 'Recall': recall, 'F1-Score': f1})
        print(f"T: {threshold:.1f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(output_dir, 'threshold_metrics_summary.csv'), index=False)
    return metrics_df

if __name__ == "__main__":
    final_results = run_zscore_scan()
    generate_report(final_results)
