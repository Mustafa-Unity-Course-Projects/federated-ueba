import os

import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import federated_ueba.task as task

# --- CONFIGURATION ---
FINAL_MODEL_PATH = "model_pickles/parameters_round_20.pkl"
DATA_PATH = "cert_data.csv"
SCALER_PATH = "scaler_data/scaler_client_0.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCAN_STRIDE = 5  # Move the window 5 rows at a time to speed up the scan


def run_zscore_scan():
    print(f"🚀 Starting Full-Timeline Z-Score Scan on {DEVICE}...")

    # 1. Load Scaler & Model
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    expected_features = list(scaler.feature_names_in_)

    # Make sure to not feed the feature to the model.
    assert not ("insider" in expected_features or "insiders" in expected_features)

    model = task.LSTMAutoencoder(input_dim=len(expected_features), hidden_dim=128).to(DEVICE)
    with open(FINAL_MODEL_PATH, "rb") as f:
        data = pickle.load(f)
        weights = data.get('global_parameters') or data.get('globa_parameters')
        model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)})
    model.eval()

    # 2. Load and Prepare Data
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.sort_values(['user', 'day'])

    # 3. Scanning Loop
    results = []
    unique_users = df['user'].unique()

    with torch.no_grad():
        for user in unique_users:
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
                std_err = 1e-6 if std_err == 0 else std_err # Prevent division by zero
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

    results_df = pd.DataFrame(results).sort_values(by="max_z_score", ascending=False)
    results_df.to_csv("federated_ueba_results.csv", index=False)
    print("💾 Results saved to federated_ueba_results.csv")
    return results_df

"""
def generate_report(df):
    threshold = 3.5 # The Z-score cutoff for an alert
    df['predicted'] = (df['max_z_score'] >= threshold).astype(float)

    y_true, y_pred = df['is_actual_insider'], df['predicted']

    print("\n" + "="*40)
    print("       FINAL PERFORMANCE METRICS")
    print("="*40)
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    print("="*40)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(df[df['is_actual_insider']==0]['max_z_score'], bins=50, alpha=0.5, label='Normal', color='gray', log=True)
    plt.hist(df[df['is_actual_insider']==1]['max_z_score'], bins=50, alpha=0.8, label='Insider', color='red', log=True)
    plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold ({threshold})')
    plt.title('Insider Detection: Z-Score Distribution')
    plt.xlabel('Z-Score (Standard Deviations)')
    plt.ylabel('User Count (Log Scale)')
    plt.legend()
    plt.show()
"""


def generate_report(df, thresholds=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], output_dir="evaluation_reports"):
    # 1. Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    metrics_data = []

    print("\n" + "=" * 55)
    print("       THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 55)

    # 2. Loop through each threshold to calculate metrics and plot
    for threshold in thresholds:
        df['predicted'] = (df['max_z_score'] >= threshold).astype(float)
        y_true, y_pred = df['is_actual_insider'], df['predicted']

        # zero_division=0 prevents warnings if a threshold flags 0 people
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        metrics_data.append({
            'Threshold': threshold,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

        print(f"Threshold: {threshold:.1f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        # 3. Visualization for this specific threshold
        plt.figure(figsize=(10, 6))
        plt.hist(df[df['is_actual_insider'] == 0]['max_z_score'], bins=50, alpha=0.5, label='Normal', color='gray',
                 log=True)
        plt.hist(df[df['is_actual_insider'] == 1]['max_z_score'], bins=50, alpha=0.8, label='Insider', color='red',
                 log=True)
        plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold ({threshold})')
        plt.title(f'Insider Detection: Z-Score Distribution (Threshold {threshold})')
        plt.xlabel('Z-Score (Standard Deviations)')
        plt.ylabel('User Count (Log Scale)')
        plt.legend()

        # 4. Save plot instead of showing
        plot_path = os.path.join(output_dir, f'zscore_distribution_t{threshold}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()  # Close figure to free up memory

    # 5. Save metrics summary to CSV
    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv_path = os.path.join(output_dir, 'threshold_metrics_summary.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\n✅ All distribution plots and metrics saved to: ./{output_dir}/")

    # 6. BONUS: Generate a single plot showing the Trade-off
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df['Threshold'], metrics_df['Precision'], marker='o', label='Precision', color='blue')
    plt.plot(metrics_df['Threshold'], metrics_df['Recall'], marker='s', label='Recall', color='orange')
    plt.plot(metrics_df['Threshold'], metrics_df['F1-Score'], marker='^', label='F1-Score', color='green')
    plt.title('Performance Metrics vs. Z-Score Threshold')
    plt.xlabel('Z-Score Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    pr_plot_path = os.path.join(output_dir, 'precision_recall_tradeoff.png')
    plt.savefig(pr_plot_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Precision-Recall tradeoff plot saved to: {pr_plot_path}")

    return metrics_df

if __name__ == "__main__":
    final_results = run_zscore_scan()
    generate_report(final_results)