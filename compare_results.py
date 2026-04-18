import os
import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import federated_ueba.task as task
from config_manager import config
from insider_detection import generate_report

# --- CONFIGURATION ---
FEDERATED_MODEL_PATH = os.path.join(config.get("federation", "save_path"), "parameters_round_20.pkl")
CENTRALIZED_MODEL_PATH = "centralized_model.pth"
CENTRALIZED_SCALER_PATH = "centralized_scaler.pkl"
DATA_PATH = config.get("data", "processed_data_path")
SCALER_DIR = config.get("data", "scaler_dir")
SCALER_FILENAME_TEMPLATE = config.get("data", "scaler_filename_template")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCAN_STRIDE = 5

def run_scan(model, scaler_dict, df, expected_features, num_clients, user_chunks):
    results = []
    with torch.no_grad():
        for client_id in range(num_clients):
            scaler = scaler_dict[client_id]
            client_users = user_chunks[client_id]
            for user in client_users:
                user_df = df[df['user'] == user]
                u_features = user_df.reindex(columns=expected_features, fill_value=0)
                u_features = u_features.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
                u_scaled = scaler.transform(u_features)
                user_tensor = torch.tensor(u_scaled, dtype=torch.float32).to(DEVICE)

                user_errors = []
                if len(user_tensor) >= task.WINDOW_SIZE:
                    for i in range(0, len(user_tensor) - task.WINDOW_SIZE + 1, SCAN_STRIDE):
                        window = user_tensor[i : i + task.WINDOW_SIZE].unsqueeze(0)
                        reconstruction = model(window)
                        error = torch.nn.functional.mse_loss(reconstruction, window).item()
                        user_errors.append(error)

                if user_errors:
                    user_errors = np.array(user_errors)
                    mean_err, std_err = np.mean(user_errors), np.std(user_errors)
                    std_err = 1e-6 if std_err == 0 else std_err
                    max_z = (np.max(user_errors) - mean_err) / std_err
                    max_raw = np.max(user_errors)
                else:
                    max_z, max_raw = 0.0, 0.0

                results.append({
                    "user": user,
                    "max_z_score": max_z,
                    "max_raw_error": max_raw,
                    "is_actual_insider": 1.0 if (user_df['insider'] == 1.0).any() else 0.0
                })
    return pd.DataFrame(results)

def compare_approaches():
    print(f"🚀 Starting Comparative Analysis on {DEVICE}...")

    # 1. Load Data
    if DATA_PATH.endswith(".pkl"):
        df = pd.read_pickle(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH, low_memory=False)
    
    time_col = 'day' if 'day' in df.columns else ('week' if 'week' in df.columns else None)
    sort_cols = ['user', time_col] if time_col else ['user']
    df = df.sort_values(sort_cols)

    # num-supernodes değerini pyproject.toml'dan çekiyoruz (num_clients settings.toml'da yok)
    num_clients = config.get_pyproject("tool", "flwr", "federations", "local-simulation", "options", "num-supernodes") or 10
    unique_users = sorted(df['user'].unique())
    user_chunks = np.array_split(unique_users, num_clients)

    # 2. Load Federated Model & Scalers
    print("🛰️ Loading Federated Approach...")
    with open(os.path.join(SCALER_DIR, SCALER_FILENAME_TEMPLATE.format(i=0)), "rb") as f:
        first_scaler = pickle.load(f)
    expected_features = list(first_scaler.feature_names_in_)

    fed_model = task.LSTMAutoencoder(input_dim=len(expected_features), hidden_dim=128).to(DEVICE)
    with open(FEDERATED_MODEL_PATH, "rb") as f:
        data = pickle.load(f)
        weights = data.get('global_parameters')
        fed_model.load_state_dict({k: torch.tensor(v) for k, v in zip(fed_model.state_dict().keys(), weights)})
    fed_model.eval()

    fed_scalers = {}
    for i in range(num_clients):
        with open(os.path.join(SCALER_DIR, SCALER_FILENAME_TEMPLATE.format(i=i)), "rb") as f:
            fed_scalers[i] = pickle.load(f)

    # 3. Load Centralized Model & Scaler
    print("🏢 Loading Centralized Approach...")
    if not os.path.exists(CENTRALIZED_MODEL_PATH):
        print(f"⚠️ Error: {CENTRALIZED_MODEL_PATH} not found. Run train_centralized.py first.")
        return

    with open(CENTRALIZED_SCALER_PATH, "rb") as f:
        cent_scaler = pickle.load(f)
    
    cent_model = task.LSTMAutoencoder(input_dim=len(expected_features), hidden_dim=128).to(DEVICE)
    cent_model.load_state_dict(torch.load(CENTRALIZED_MODEL_PATH, map_location=DEVICE))
    cent_model.eval()

    # Centralized approach uses ONE scaler for ALL users
    cent_scalers = {i: cent_scaler for i in range(num_clients)}

    # 4. Run Scans
    print("🔍 Scanning Federated results...")
    fed_results = run_scan(fed_model, fed_scalers, df, expected_features, num_clients, user_chunks)
    
    print("🔍 Scanning Centralized results...")
    cent_results = run_scan(cent_model, cent_scalers, df, expected_features, num_clients, user_chunks)

    # 5. Generate Reports
    print("\n--- Federated Report ---")
    fed_metrics = generate_report(fed_results, output_dir="comparison_reports/federated")
    
    print("\n--- Centralized Report ---")
    cent_metrics = generate_report(cent_results, output_dir="comparison_reports/centralized")

    # 6. Final Summary
    comparison_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score'],
        'Federated (T=3.5)': [
            fed_metrics.loc[fed_metrics['Threshold'] == 3.5, 'Precision'].values[0],
            fed_metrics.loc[fed_metrics['Threshold'] == 3.5, 'Recall'].values[0],
            fed_metrics.loc[fed_metrics['Threshold'] == 3.5, 'F1-Score'].values[0]
        ],
        'Centralized (T=3.5)': [
            cent_metrics.loc[cent_metrics['Threshold'] == 3.5, 'Precision'].values[0],
            cent_metrics.loc[cent_metrics['Threshold'] == 3.5, 'Recall'].values[0],
            cent_metrics.loc[cent_metrics['Threshold'] == 3.5, 'F1-Score'].values[0]
        ]
    })

    print("\n" + "="*40)
    print("       FINAL COMPARISON SUMMARY (T=3.5)")
    print("="*40)
    print(comparison_df.to_string(index=False))
    print("="*40)
    
    comparison_df.to_csv("approach_comparison.csv", index=False)
    print("💾 Final comparison saved to approach_comparison.csv")

if __name__ == "__main__":
    compare_approaches()
