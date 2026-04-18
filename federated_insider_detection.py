import os
import shutil
import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
import toml
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             average_precision_score, precision_recall_curve, 
                             accuracy_score, balanced_accuracy_score, confusion_matrix)
import argparse
import subprocess
import time
from datetime import datetime

# --- 1. RUN ID ORCHESTRATION ---
# We must determine the RUN_ID BEFORE importing any modules that use 'config'

BASE_REPORT_DIR = "federated_evaluation_reports"

def get_numeric_runs():
    """Returns a sorted list of existing numeric run IDs."""
    if not os.path.exists(BASE_REPORT_DIR):
        return []
    runs = []
    for entry in os.listdir(BASE_REPORT_DIR):
        if os.path.isdir(os.path.join(BASE_REPORT_DIR, entry)) and entry.isdigit():
            runs.append(int(entry))
    return sorted(runs)

def get_next_run_id():
    runs = get_numeric_runs()
    return str(runs[-1] + 1) if runs else "0"

def get_latest_run_id():
    runs = get_numeric_runs()
    return str(runs[-1]) if runs else "0"

# Peek at CLI args to decide on RUN_ID logic
parser = argparse.ArgumentParser(description="F-UEBA Unified Experiment Manager", add_help=False)
parser.add_argument("--mode", choices=["full", "train", "eval"], default="full")
parser.add_argument("--run_id", type=str, help="Specific Run ID to use/evaluate")
args, unknown = parser.parse_known_args()

if args.run_id:
    current_run_id = args.run_id
elif args.mode == "eval":
    current_run_id = get_latest_run_id()
else:
    current_run_id = get_next_run_id()

# Inject into environment for all sub-processes (Server/Clients)
os.environ["RUN_ID"] = current_run_id

# --- 2. CONFIG & TASK IMPORTS ---
from config_manager import config
import federated_ueba.task as task

# --- CONFIGURATION ---
SAVE_PATH = config.get("federation", "save_path")
DATA_PATH = config.get("data", "processed_data_path")
SCALER_DIR = config.get("data", "scaler_dir")
SCALER_FILENAME_TEMPLATE = config.get("data", "scaler_filename_template")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPORT_DIR = os.path.join(BASE_REPORT_DIR, current_run_id)
ROUND_RESULTS_DIR = os.path.join(REPORT_DIR, "round_by_round_results")

HIDDEN_DIM = config.get("model", "hidden_dim") or 128
TOP_K_FEATURES = 5
PERSISTENCE_WINDOW = 3 
DIVERSITY_THRESHOLD = 2.0 
SCAN_STRIDE = 1

def cleanup_old_metrics():
    """Removes previous evaluation metrics and reports for the current run_id."""
    if os.path.exists(REPORT_DIR):
        print(f"🧹 Refreshing report directory: {REPORT_DIR}")
        shutil.rmtree(REPORT_DIR)
    os.makedirs(ROUND_RESULTS_DIR, exist_ok=True)

def perform_scan(model, df, user_chunks, expected_features, num_clients):
    results = []
    with torch.no_grad():
        for client_id in range(num_clients):
            scaler_path = os.path.join(SCALER_DIR, SCALER_FILENAME_TEMPLATE.format(i=client_id))
            stats_path = os.path.join(SCALER_DIR, f"error_stats_client_{client_id}.pkl")
            if not os.path.exists(scaler_path): continue
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            
            if os.path.exists(stats_path):
                with open(stats_path, "rb") as f:
                    calib = pickle.load(f)
                    mean_per_feature, std_per_feature = calib["mean_per_feature"], calib["std_per_feature"]
            else:
                mean_per_feature, std_per_feature = np.full(len(expected_features), 0.02), np.full(len(expected_features), 0.05)

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
                        sq_err = torch.mean((reconstruction - window)**2, dim=1).squeeze(0).cpu().numpy()
                        feat_z = (sq_err - mean_per_feature) / (std_per_feature + 1e-6)
                        pos_feat_z = np.maximum(feat_z, 0)
                        top_k_z = np.sort(pos_feat_z)[-TOP_K_FEATURES:]
                        diversity_factor = 1.0 + (np.sum(pos_feat_z > DIVERSITY_THRESHOLD) / len(expected_features))
                        user_window_metrics.append(np.mean(top_k_z) * diversity_factor)

                has_insider = (user_df['insider'] != 0).any()
                if user_window_metrics:
                    arr = np.sort(np.array(user_window_metrics))
                    final_score = np.mean(arr[-min(len(arr), PERSISTENCE_WINDOW):])
                else: final_score = 0.0
                results.append({"user": user, "max_z_score": final_score, "is_actual_insider": 1.0 if has_insider else 0.0})
    return pd.DataFrame(results)

def calculate_metrics_at_threshold(y_true, y_scores, threshold):
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }

def run_evaluation():
    # Note: We don't cleanup_old_metrics() here because training might have already put logs there.
    # Instead, we ensure the directories exist.
    os.makedirs(ROUND_RESULTS_DIR, exist_ok=True)
    
    print(f"📊 Starting Evaluation for run '{current_run_id}'...")
    
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.sort_values(['user', 'day' if 'day' in df.columns else 'week'])
    num_clients = config.get_pyproject("tool", "flwr", "federations", "local-simulation", "options", "num-supernodes") or 10
    user_chunks = np.array_split(sorted(df['user'].unique()), num_clients)

    first_scaler_path = os.path.join(SCALER_DIR, SCALER_FILENAME_TEMPLATE.format(i=0))
    if not os.path.exists(first_scaler_path):
        print(f"❌ Error: Scaler file not found at {first_scaler_path}. Check if training finished correctly.")
        return
        
    with open(first_scaler_path, "rb") as f:
        expected_features = list(pickle.load(f).feature_names_in_)

    model = task.LSTMAutoencoder(input_dim=len(expected_features), hidden_dim=HIDDEN_DIM).to(DEVICE)
    candidates = []
    if os.path.exists(SAVE_PATH):
        files = [f for f in os.listdir(SAVE_PATH) if f.startswith("parameters_round_") and f.endswith(".pkl")]
        for f in files:
            r_num = int(f.split('_')[-1].split('.')[0])
            if r_num % 10 == 0:
                candidates.append((r_num, os.path.join(SAVE_PATH, f)))
    candidates = sorted(candidates, key=lambda x: x[0])
    
    summary_data = []
    best_pr_auc, best_round, best_results = 0, -1, None

    comm_log_path = os.path.join(REPORT_DIR, f"communication_log.csv")
    avg_comm_per_round = pd.read_csv(comm_log_path, header=None)[1].mean() if os.path.exists(comm_log_path) else 0.0

    for r_num, r_path in candidates:
        print(f"🔄 Evaluating Round {r_num}...")
        with open(r_path, "rb") as f:
            data = pickle.load(f)
            weights = data.get('global_parameters')
            state_dict_keys = model.state_dict().keys()
            model.load_state_dict({k: torch.tensor(w) for k, w in zip(state_dict_keys, weights)})
        model.eval()
        
        current_results = perform_scan(model, df, user_chunks, expected_features, num_clients)
        y_true, y_scores = current_results['is_actual_insider'], current_results['max_z_score']
        
        pr_auc_val = average_precision_score(y_true, y_scores)
        precision_vals, recall_vals, thresholds_vals = precision_recall_curve(y_true, y_scores)
        f1_scores = np.where((precision_vals + recall_vals) > 0, (2 * precision_vals * recall_vals) / (precision_vals + recall_vals), 0)
        best_f1_idx = np.argmax(f1_scores)
        opt_threshold = thresholds_vals[best_f1_idx] if best_f1_idx < len(thresholds_vals) else thresholds_vals[-1]
        
        m = calculate_metrics_at_threshold(y_true, y_scores, opt_threshold)
        summary_data.append({
            "Round": r_num, "PR-AUC": pr_auc_val, "Max-F1": m["f1"],
            "Precision": m["precision"], "Recall": m["recall"], 
            "Accuracy": m["accuracy"], "Balanced_Accuracy": m["balanced_accuracy"],
            "TP": m["tp"], "FP": m["fp"], "TN": m["tn"], "FN": m["fn"],
            "Optimal-Threshold": opt_threshold
        })
        current_results.to_csv(os.path.join(ROUND_RESULTS_DIR, f"round_{r_num}_results.csv"), index=False)

        if pr_auc_val > best_pr_auc:
            best_pr_auc, best_round, best_results = pr_auc_val, r_num, current_results

    if best_results is not None:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(REPORT_DIR, "federated_rounds_comparison.csv"), index=False)
        
        # Save Experiment Summary
        best_row = summary_df.loc[summary_df['PR-AUC'].idxmax()]
        experiment_summary = {
            "run_id": current_run_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": config._settings,
            "best_metrics": best_row.to_dict(),
            "avg_communication_mb": avg_comm_per_round
        }
        with open(os.path.join(REPORT_DIR, "experiment_summary.json"), "w") as f:
            json.dump(experiment_summary, f, indent=4)
        
        generate_plots(best_results, best_round, best_pr_auc, summary_df, avg_comm_per_round)

def generate_plots(df, best_round, pr_auc_val, summary_df, avg_comm_per_round):
    # Honest Learning Curve
    plt.figure(figsize=(10, 6))
    plt.plot(summary_df['Round'], summary_df['PR-AUC'], marker='o', label='PR-AUC', color='blue')
    plt.plot(summary_df['Round'], summary_df['Balanced_Accuracy'], marker='^', label='Balanced Accuracy', color='purple')
    plt.plot(summary_df['Round'], summary_df['Max-F1'], marker='s', label='Max F1-Score', color='green')
    plt.title(f'FL Progress: Run {current_run_id}')
    plt.xlabel('Round Number'); plt.ylabel('Score'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(REPORT_DIR, 'learning_progress.png'))
    plt.close()

    # Threshold Sensitivity
    y_true, y_scores = df['is_actual_insider'], df['max_z_score']
    thresholds = np.linspace(0, np.max(y_scores), 100)
    prec_sweep, rec_sweep, bacc_sweep = [], [], []
    for t in thresholds:
        m = calculate_metrics_at_threshold(y_true, y_scores, t)
        prec_sweep.append(m["precision"]); rec_sweep.append(m["recall"]); bacc_sweep.append(m["balanced_accuracy"])
        
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, prec_sweep, label='Precision')
    plt.plot(thresholds, rec_sweep, label='Recall')
    plt.plot(thresholds, bacc_sweep, label='Balanced Accuracy')
    plt.title(f'Threshold Analysis: Run {current_run_id}')
    plt.xlabel('Z-Score Threshold'); plt.ylabel('Score'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(REPORT_DIR, 'zscore_cutoff_analysis.png'))
    plt.close()
    
    print(f"\n✅ Evaluation Complete for Run {current_run_id}. Best Round: {best_round}. PR-AUC: {pr_auc_val:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Manager")
    parser.add_argument("--mode", choices=["full", "train", "eval"], default="full")
    parser.add_argument("--run_id", type=str, help="Specific run ID")
    args = parser.parse_args()

    if args.mode in ["full", "train"]:
        cleanup_old_metrics()
        # Save run config immediately
        with open(os.path.join(REPORT_DIR, "run_config.toml"), "w") as f:
            toml.dump(config._settings, f)

        num_supernodes = config.get_pyproject("tool", "flwr", "federations", "local-simulation", "options", "num-supernodes") or 10
        print(f"🚀 Starting Federated Training (Run {current_run_id})...")
        command = ["flower-simulation", "--app", ".", "--num-supernodes", str(num_supernodes)]
        subprocess.run(command, check=True)
    
    if args.mode in ["full", "eval"]:
        run_evaluation()
