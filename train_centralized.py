import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import federated_ueba.task as task
from config_manager import config
import matplotlib.pyplot as plt
import json
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             average_precision_score, precision_recall_curve, 
                             accuracy_score, balanced_accuracy_score, confusion_matrix)
import random # Import random module

# --- CONFIGURATION ---
DATA_PATH = config.get("data", "processed_data_path")
CENTRALIZED_MODEL_PATH = "centralized_model.pth"
CENTRALIZED_SCALER_PATH = "centralized_scaler.pkl"
CENTRALIZED_ERROR_STATS_PATH = "centralized_error_stats.pkl" # Defined path for error stats
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = config.get("model", "learning_rate") or 0.001
SELECTED_FEATURES = config.get("data", "selected_features") or None
EARLY_STOPPING_PATIENCE = 10 
HIDDEN_DIM = config.get("model", "hidden_dim") or 64
TOP_K_FEATURES = config.get("anomaly_detection", "top_k_features") or 5 # From config
PERSISTENCE_WINDOW = config.get("anomaly_detection", "persistence_window") or 3 # From config
DIVERSITY_THRESHOLD = config.get("anomaly_detection", "diversity_threshold") or 2.0 # From config
SCAN_STRIDE = config.get("anomaly_detection", "scan_stride") or 1 # From config
INFERENCE_BATCH_SIZE = config.get("data", "batch_size") or 128 # From config
RANDOM_SEED = 42 # Define a random seed for reproducibility

# Reporting Configuration
CENTRALIZED_REPORT_DIR = "centralized_evaluation_reports"
os.makedirs(CENTRALIZED_REPORT_DIR, exist_ok=True)

def calculate_metrics_at_threshold(y_true, y_scores, threshold):
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }

def perform_centralized_scan(model, df_full, scaler, error_stats, features):
    results = []
    model.eval() # Ensure model is in evaluation mode
    
    mean_per_feature = error_stats["mean_per_feature"]
    std_per_feature = error_stats["std_per_feature"]

    with torch.no_grad():
        unique_users = df_full['user'].unique()
        for user in unique_users:
            user_df = df_full[df_full['user'] == user].copy()
            u_features = user_df.reindex(columns=features, fill_value=0)
            u_features = u_features.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
            u_features = np.log1p(u_features.clip(lower=0))
            
            u_scaled = scaler.transform(u_features)
            user_tensor = torch.tensor(u_scaled, dtype=torch.float32).to(DEVICE)

            user_window_metrics = []
            
            if len(user_tensor) >= task.WINDOW_SIZE:
                windows_to_process = []
                for i in range(0, len(user_tensor) - task.WINDOW_SIZE + 1, SCAN_STRIDE):
                    windows_to_process.append(user_tensor[i : i + task.WINDOW_SIZE])

                if not windows_to_process:
                    final_score = 0.0
                else:
                    all_window_errors = []
                    for i in range(0, len(windows_to_process), INFERENCE_BATCH_SIZE):
                        batch_windows = torch.stack(windows_to_process[i : i + INFERENCE_BATCH_SIZE]).to(DEVICE)
                        reconstruction_batch = model(batch_windows)
                        
                        sq_err_batch = torch.mean((reconstruction_batch - batch_windows)**2, dim=1).cpu().numpy()
                        
                        for sq_err in sq_err_batch:
                            feat_z = (sq_err - mean_per_feature) / (std_per_feature + 1e-6)
                            pos_feat_z = np.maximum(feat_z, 0)
                            top_k_z = np.sort(pos_feat_z)[-TOP_K_FEATURES:]
                            diversity_factor = 1.0 + (np.sum(pos_feat_z > DIVERSITY_THRESHOLD) / len(features))
                            all_window_errors.append(np.mean(top_k_z) * diversity_factor)
                    
                    user_window_metrics = all_window_errors

                    if user_window_metrics:
                        arr = np.sort(np.array(user_window_metrics))
                        final_score = np.mean(arr[-min(len(arr), PERSISTENCE_WINDOW):])
                    else:
                        final_score = 0.0
            else: 
                final_score = 0.0

            has_insider = (user_df['insider'] != 0).any() if 'insider' in user_df.columns else 0.0
            results.append({"user": user, "max_z_score": final_score, "is_actual_insider": 1.0 if has_insider else 0.0})
    return pd.DataFrame(results)

def calculate_error_stats(model, val_loader, features):
    """Calculates error statistics (mean and std per feature) from the validation set."""
    model.eval()
    all_feature_errors = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            output = model(batch)
            batch_feature_error = torch.mean(torch.abs(output - batch), dim=1) # (B, D)
            all_feature_errors.append(batch_feature_error.cpu().numpy())

    all_feature_errors = np.concatenate(all_feature_errors, axis=0)
    
    # TRIMMED STATS: Ignore the top 1% of noisy validation points to make anomalies pop more
    trim_pct = 1
    trim_idx = int(len(all_feature_errors) * (1 - trim_pct/100))
    mean_per_feature = np.zeros(len(features))
    std_per_feature = np.zeros(len(features))
    
    for d in range(len(features)):
        col = all_feature_errors[:, d]
        trimmed_col = np.sort(col)[:trim_idx]
        mean_per_feature[d] = np.mean(trimmed_col)
        std_per_feature[d] = np.std(trimmed_col)

    # Global Distribution calculation for the combined Top-K metric
    topk_window_metrics = []
    for i in range(len(all_feature_errors)):
        feat_z = (all_feature_errors[i] - mean_per_feature) / (std_per_feature + 1e-6)
        topk_metric = np.mean(np.sort(np.maximum(feat_z, 0))[-TOP_K_FEATURES:])
        topk_window_metrics.append(topk_metric)

    error_stats = {
        "mean_per_feature": mean_per_feature,
        "std_per_feature": std_per_feature,
        "topk_metric_mean": np.mean(topk_window_metrics),
        "topk_metric_std": np.std(topk_window_metrics)
    }
    return error_stats

def evaluate_anomaly_detection(model, df_full, scaler, error_stats, features):
    """Evaluates anomaly detection performance and returns key metrics."""
    centralized_results_df = perform_centralized_scan(model, df_full, scaler, error_stats, features)
    
    y_true = centralized_results_df['is_actual_insider']
    y_scores = centralized_results_df['max_z_score']

    pr_auc_val = float(average_precision_score(y_true, y_scores))
    precision_vals, recall_vals, thresholds_vals = precision_recall_curve(y_true, y_scores)
    
    numerator = 2 * precision_vals * recall_vals
    denominator = precision_vals + recall_vals
    f1_scores = np.zeros_like(numerator)
    non_zero_denominator_mask = denominator > 0
    f1_scores[non_zero_denominator_mask] = numerator[non_zero_denominator_mask] / denominator[non_zero_denominator_mask]

    best_f1_idx = np.argmax(f1_scores)
    opt_threshold = float(thresholds_vals[best_f1_idx]) if best_f1_idx < len(thresholds_vals) else float(thresholds_vals[-1])
    
    best_metrics = calculate_metrics_at_threshold(y_true, y_scores, opt_threshold)
    
    return {
        "pr_auc": pr_auc_val,
        "best_f1": best_metrics['f1'],
        "optimal_threshold": opt_threshold,
        "precision_at_best_f1": best_metrics["precision"],
        "recall_at_best_f1": best_metrics["recall"],
        "accuracy_at_best_f1": best_metrics["accuracy"],
        "balanced_accuracy_at_best_f1": best_metrics["balanced_accuracy"],
        "tp_at_best_f1": best_metrics["tp"],
        "fp_at_best_f1": best_metrics["fp"],
        "tn_at_best_f1": best_metrics["tn"],
        "fn_at_best_f1": best_metrics["fn"],
    }


def train_centralized():
    print(f"🚀 Starting Bidirectional Centralized Model Training on {DEVICE}...")

    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED) # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"⚙️ Random seed set to {RANDOM_SEED} for reproducibility.")

    # 1. Load Data (Full dataset for feature extraction, then filtered for training)
    df_full = pd.read_csv(DATA_PATH) # Load full dataset for later evaluation

    # 2. Dynamic Feature Selection
    metadata = ['user', 'day', 'week', 'pc', 'activity', 'id', 'label', 'insider', 'to', 'from', 'starttime', 'endtime', 'pcid', 'time_stamp', 'actid']
    features = [c for c in SELECTED_FEATURES if c in df_full.columns] if SELECTED_FEATURES else [c for c in df_full.columns if c not in metadata]

    print(f"📊 Training with {len(features)} features.")

    # 3. Filter Training Data (Normal only)
    df_train = df_full[df_full['insider'] == 0].copy() if 'insider' in df_full.columns else df_full.copy()
    
    # 4. Numeric Conversion & Scaling
    df_train[features] = df_train[features].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    df_train[features] = np.log1p(df_train[features].clip(lower=0))
    scaler = StandardScaler()
    df_train[features] = scaler.fit_transform(df_train[features])

    with open(CENTRALIZED_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler saved to {CENTRALIZED_SCALER_PATH}")

    # 5. Sequence Building
    unique_users = df_train['user'].unique()
    train_users, val_users = train_test_split(unique_users, test_size=0.2, random_state=RANDOM_SEED) # Use RANDOM_SEED here

    def get_sequences(user_list, dataframe): # Modified to accept dataframe
        seqs = []
        for user in user_list:
            user_data = dataframe[dataframe['user'] == user].sort_values('day')[features].values
            if len(user_data) >= task.WINDOW_SIZE:
                for i in range(0, len(user_data) - task.WINDOW_SIZE + 1, task.STRIDE):
                    seqs.append(user_data[i:i + task.WINDOW_SIZE])
        return seqs

    train_seqs = get_sequences(train_users, df_train)
    val_seqs = get_sequences(val_users, df_train)
    train_loader = torch.utils.data.DataLoader(torch.tensor(np.array(train_seqs), dtype=torch.float32), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.tensor(np.array(val_seqs), dtype=torch.float32), batch_size=BATCH_SIZE, shuffle=False)

    # 6. Model & Optimization
    model = task.LSTMAutoencoder(input_dim=len(features), hidden_dim=HIDDEN_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    training_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                output = model(batch)
                val_loss += criterion(output, batch).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate error stats and anomaly detection metrics for the current epoch
        current_error_stats = calculate_error_stats(model, val_loader, features)
        current_ad_metrics = evaluate_anomaly_detection(model, df_full, scaler, current_error_stats, features)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | F1: {current_ad_metrics['best_f1']:.4f} | PR-AUC: {current_ad_metrics['pr_auc']:.4f}")
        
        training_history.append({
            "Epoch": epoch + 1,
            "Train Loss": avg_train_loss,
            "Val Loss": avg_val_loss,
            "F1_Score": current_ad_metrics['best_f1'],
            "PR_AUC": current_ad_metrics['pr_auc']
        })

        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            # Save the best error stats along with the best model
            best_error_stats = current_error_stats
            final_ad_metrics = current_ad_metrics # Store the metrics for the best model
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), CENTRALIZED_MODEL_PATH)
    print(f"💾 Model saved to {CENTRALIZED_MODEL_PATH}")
    
    # Save the error stats corresponding to the best model
    with open(CENTRALIZED_ERROR_STATS_PATH, "wb") as f:
        pickle.dump(best_error_stats, f)
    print(f"💾 Optimized Stats saved to {CENTRALIZED_ERROR_STATS_PATH}")

    # --- Reporting Section ---
    training_df = pd.DataFrame(training_history)
    training_df.to_csv(os.path.join(CENTRALIZED_REPORT_DIR, "centralized_training_progress.csv"), index=False)
    print(f"💾 Training progress saved to {os.path.join(CENTRALIZED_REPORT_DIR, 'centralized_training_progress.csv')}")

    plt.figure(figsize=(10, 6))
    plt.plot(training_df['Epoch'], training_df['Train Loss'], label='Train Loss')
    plt.plot(training_df['Epoch'], training_df['Val Loss'], label='Validation Loss')
    plt.title('Centralized Model Learning Progress')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CENTRALIZED_REPORT_DIR, 'centralized_learning_progress.png'))
    plt.close()
    print(f"💾 Learning progress plot saved to {os.path.join(CENTRALIZED_REPORT_DIR, 'centralized_learning_progress.png')}")

    # New: Plot for F1 Score and PR-AUC over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(training_df['Epoch'], training_df['F1_Score'], label='F1 Score', marker='o', markersize=4)
    plt.plot(training_df['Epoch'], training_df['PR_AUC'], label='PR-AUC', marker='x', markersize=4)
    plt.title('Centralized Anomaly Detection Metrics Over Epochs')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.ylim(0, 1) # Metrics are typically between 0 and 1
    plt.savefig(os.path.join(CENTRALIZED_REPORT_DIR, 'centralized_ad_metrics_by_epoch.png'))
    plt.close()
    print(f"💾 Anomaly detection metrics by epoch plot saved to {os.path.join(CENTRALIZED_REPORT_DIR, 'centralized_ad_metrics_by_epoch.png')}")


    # Plot for Final Anomaly Detection Metrics (using the metrics from the best model)
    metrics_names = ['PR-AUC', 'Max-F1', 'Balanced Accuracy']
    metrics_values = [final_ad_metrics['pr_auc'], final_ad_metrics['best_f1'], final_ad_metrics['balanced_accuracy_at_best_f1']]

    plt.figure(figsize=(8, 6))
    plt.bar(metrics_names, metrics_values, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.ylim(0, 1) # Metrics are typically between 0 and 1
    plt.title('Centralized Final Anomaly Detection Metrics (Best Model)')
    plt.ylabel('Score')
    for i, value in enumerate(metrics_values):
        plt.text(i, value + 0.02, f'{value:.4f}', ha='center', va='bottom')
    plt.savefig(os.path.join(CENTRALIZED_REPORT_DIR, 'centralized_final_metrics.png'))
    plt.close()
    print(f"💾 Final anomaly detection metrics plot saved to {os.path.join(CENTRALIZED_REPORT_DIR, 'centralized_final_metrics.png')}")

    summary_data = {
        "model_path": CENTRALIZED_MODEL_PATH,
        "scaler_path": CENTRALIZED_SCALER_PATH,
        "error_stats_path": CENTRALIZED_ERROR_STATS_PATH,
        "best_validation_loss": float(best_val_loss),
        "epochs_trained": int(len(training_history)),
        "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
        "epochs_config": int(EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "learning_rate": float(LEARNING_RATE),
        "hidden_dim": int(HIDDEN_DIM),
        "input_features_count": int(len(features)),
        "window_size": int(task.WINDOW_SIZE),
        "stride": int(task.STRIDE),
        "device": str(DEVICE),
        "anomaly_detection_metrics": final_ad_metrics # Use the metrics from the best model
    }
    with open(os.path.join(CENTRALIZED_REPORT_DIR, "centralized_experiment_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"💾 Experiment summary saved to {os.path.join(CENTRALIZED_REPORT_DIR, 'centralized_experiment_summary.json')}")

if __name__ == "__main__":
    train_centralized()