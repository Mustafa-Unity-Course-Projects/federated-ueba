import os
import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import federated_ueba.task as task
from config_manager import config

# --- CONFIGURATION ---
SAVE_PATH = config.get("federation", "save_path")
DATA_PATH = config.get("data", "processed_data_path")
SCALER_DIR = config.get("data", "scaler_dir")
SCALER_FILENAME_TEMPLATE = config.get("data", "scaler_filename_template")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPORT_DIR = "federated_evaluation_reports"

HIDDEN_DIM = config.get("model", "hidden_dim") or 64
TOP_K_FEATURES = 5
PERSISTENCE_WINDOW = 3 
DIVERSITY_THRESHOLD = 2.0 
SCAN_STRIDE = 1

def get_latest_model_path():
    if not os.path.exists(SAVE_PATH): return None
    files = [f for f in os.listdir(SAVE_PATH) if f.startswith("parameters_round_") and f.endswith(".pkl")]
    if not files: return None
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
    return os.path.join(SAVE_PATH, files[0])

def run_federated_scan():
    print(f"🚀 Starting Federated Multi-Vector Scan on {DEVICE}...")
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 1. Load Data
    df = pd.read_csv(DATA_PATH, low_memory=False)
    time_col = 'day' if 'day' in df.columns else ('week' if 'week' in df.columns else None)
    df = df.sort_values(['user', time_col])

    num_clients = config.get_pyproject("tool", "flwr", "federations", "local-simulation", "options", "num-supernodes") or 10
    unique_users = sorted(df['user'].unique())
    user_chunks = np.array_split(unique_users, num_clients)

    # 2. Setup Model
    first_scaler_path = os.path.join(SCALER_DIR, SCALER_FILENAME_TEMPLATE.format(i=0))
    with open(first_scaler_path, "rb") as f:
        scaler = pickle.load(f)
    expected_features = list(scaler.feature_names_in_)

    model = task.LSTMAutoencoder(input_dim=len(expected_features), hidden_dim=HIDDEN_DIM).to(DEVICE)
    model_path = get_latest_model_path()
    
    print(f"📦 Loading Global Model: {os.path.basename(model_path)}")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
        weights = data.get('global_parameters') or data.get('globa_parameters')
        state_dict_keys = model.state_dict().keys()
        new_state_dict = {k: torch.tensor(w) for k, w in zip(state_dict_keys, weights)}
        model.load_state_dict(new_state_dict)
    model.eval()

    # 3. Scanning Loop
    results = []
    
    with torch.no_grad():
        for client_id in range(num_clients):
            scaler_path = os.path.join(SCALER_DIR, SCALER_FILENAME_TEMPLATE.format(i=client_id))
            if not os.path.exists(scaler_path): continue

            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            
            client_users = user_chunks[client_id]
            for user in client_users:
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
                        
                        abs_err = torch.mean(torch.abs(reconstruction - window), dim=1).squeeze(0).cpu().numpy()
                        
                        # Federated local Z-score baseline (synced with central)
                        feat_z = (abs_err - 0.02) / 0.05 
                        pos_feat_z = np.maximum(feat_z, 0)
                        top_k_z = np.sort(pos_feat_z)[-TOP_K_FEATURES:]
                        
                        num_anomalous = np.sum(pos_feat_z > DIVERSITY_THRESHOLD)
                        diversity_factor = 1.0 + (num_anomalous / len(expected_features))
                        
                        user_window_metrics.append(np.mean(top_k_z) * diversity_factor)

                has_insider = (user_df['insider'] != 0).any()
                if user_window_metrics:
                    user_window_metrics_arr = np.sort(np.array(user_window_metrics))
                    final_score = np.mean(user_window_metrics_arr[-min(len(user_window_metrics_arr), PERSISTENCE_WINDOW):])
                else:
                    final_score = 0.0

                results.append({
                    "user": user,
                    "max_z_score": final_score,
                    "is_actual_insider": 1.0 if has_insider else 0.0
                })

    results_df = pd.DataFrame(results).sort_values(by="max_z_score", ascending=False)
    results_df.to_csv("federated_insider_results_advanced.csv", index=False)
    print(f"✅ Scan complete. Results saved.")
    return results_df

def generate_report(df):
    print("\n📊 Generating Federated Performance Report...")
    
    # 1. Z-Score Distribution Histogram
    plt.figure(figsize=(12, 6))
    plt.hist(df[df['is_actual_insider'] == 0]['max_z_score'], bins=100, alpha=0.5, label='Normal Users', color='gray', log=True)
    plt.hist(df[df['is_actual_insider'] == 1]['max_z_score'], bins=100, alpha=0.8, label='Insider Threats', color='red', log=True)
    plt.title('Federated Z-Score Distribution (Log Scale)')
    plt.xlabel('Detection Score')
    plt.ylabel('User Count')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(REPORT_DIR, 'z_score_distribution.png'))
    plt.close()

    # 2. Precision-Recall Trade-off
    thresholds = np.linspace(df['max_z_score'].min(), df['max_z_score'].max(), 100)
    p_scores, r_scores, f1_scores = [], [], []
    
    for t in thresholds:
        y_true = df['is_actual_insider']
        y_pred = (df['max_z_score'] >= t).astype(float)
        p_scores.append(precision_score(y_true, y_pred, zero_division=0))
        r_scores.append(recall_score(y_true, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, p_scores, label='Precision', color='blue', lw=2)
    plt.plot(thresholds, r_scores, label='Recall', color='orange', lw=2)
    plt.plot(thresholds, f1_scores, label='F1-Score', color='green', lw=2, linestyle='--')
    plt.title('Performance Metrics vs. Detection Threshold')
    plt.xlabel('Detection Threshold (Z-Score)')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(REPORT_DIR, 'threshold_performance.png'))
    plt.close()

    # 3. AUC-ROC and PR Curve
    auc_roc = roc_auc_score(df['is_actual_insider'], df['max_z_score'])
    
    # Text Report Summary
    best_f1_idx = np.argmax(f1_scores)
    best_t = thresholds[best_f1_idx]
    
    print("\n" + "=" * 55)
    print(f"       FEDERATED EVALUATION SUMMARY (AUC-ROC: {auc_roc:.4f})")
    print("=" * 55)
    print(f"Optimal Threshold (Max F1): {best_t:.2f}")
    print(f"Precision at Optimal:       {p_scores[best_f1_idx]:.4f}")
    print(f"Recall at Optimal:          {r_scores[best_f1_idx]:.4f}")
    print(f"F1-Score at Optimal:        {f1_scores[best_f1_idx]:.4f}")
    print("=" * 55)

    # Save summary stats
    metrics_summary = pd.DataFrame({
        'Threshold': thresholds,
        'Precision': p_scores,
        'Recall': r_scores,
        'F1-Score': f1_scores
    })
    metrics_summary.to_csv(os.path.join(REPORT_DIR, 'full_metrics_sweep.csv'), index=False)
    print(f"✅ All visualizations and data saved to: ./{REPORT_DIR}/")

if __name__ == "__main__":
    res = run_federated_scan()
    generate_report(res)
