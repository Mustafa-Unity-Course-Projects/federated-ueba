import os
import shutil
import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             average_precision_score, precision_recall_curve, 
                             accuracy_score, balanced_accuracy_score, confusion_matrix)
import argparse
import subprocess
import time
from datetime import datetime

# --- 1. RUN ID ORCHESTRATION ---
BASE_REPORT_DIR = "federated_evaluation_reports"

def run_single_experiment(experiment_name, mode):
    # Inject into environment so config_manager picks it up
    os.environ["EXPERIMENT_NAME"] = experiment_name
    
    # Reload/Re-initialize config
    # This import needs to be here to ensure config_manager re-reads env vars
    import config_manager
    from config_manager import config
    config.set_experiment(experiment_name)
    
    import federated_ueba.task as task

    # --- CONFIGURATION ---
    SAVE_PATH = config.get("federation", "save_path")
    DATA_PATH = config.get("data", "processed_data_path")
    SCALER_DIR = config.get("data", "scaler_dir")
    SCALER_FILENAME_TEMPLATE = config.get("data", "scaler_filename_template")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    REPORT_DIR = os.path.join(BASE_REPORT_DIR, experiment_name)
    ROUND_RESULTS_DIR = os.path.join(REPORT_DIR, "round_by_round_results")

    HIDDEN_DIM = config.get("model", "hidden_dim") or 128
    TOP_K_FEATURES = config.get("anomaly_detection", "top_k_features") or 5
    PERSISTENCE_WINDOW = config.get("anomaly_detection", "persistence_window") or 3
    DIVERSITY_THRESHOLD = config.get("anomaly_detection", "diversity_threshold") or 2.0
    SCAN_STRIDE = config.get("anomaly_detection", "scan_stride") or 1
    INFERENCE_BATCH_SIZE = config.get("data", "batch_size") or 128 # Re-using batch_size from task.py

    # --- Check if experiment is already done ---
    summary_file_path = os.path.join(REPORT_DIR, "experiment_summary.json")
    if os.path.exists(summary_file_path):
        try:
            with open(summary_file_path, "r") as f:
                summary_data = json.load(f)
            if summary_data.get("done", False) and mode != "train": # Only skip if done AND not explicitly training
                print(f"✅ Experiment '{experiment_name}' already completed. Skipping.")
                return
        except json.JSONDecodeError:
            print(f"⚠️ Corrupted experiment_summary.json for '{experiment_name}'. Re-running.")
            # If corrupted, proceed to re-run and overwrite

    def cleanup_old_metrics():
        # Always clean up model and scaler directories for the current experiment
        # to ensure a fresh start for weights and scalers.
        if os.path.exists(SAVE_PATH):
            print(f"🧹 Refreshing model save path: {SAVE_PATH}")
            shutil.rmtree(SAVE_PATH)
        if os.path.exists(SCALER_DIR):
            print(f"🧹 Refreshing scaler data path: {SCALER_DIR}")
            shutil.rmtree(SCALER_DIR)

        # Conditional cleanup for report directory:
        # Only clean if the summary file doesn't exist or the previous run was incomplete.
        current_summary_data = {}
        if os.path.exists(summary_file_path):
            try:
                with open(summary_file_path, "r") as f:
                    current_summary_data = json.load(f)
            except json.JSONDecodeError:
                pass # Treat as corrupted, will be overwritten

        if not os.path.exists(summary_file_path) or not current_summary_data.get("done", False):
            if os.path.exists(REPORT_DIR):
                print(f"🧹 Refreshing report directory: {REPORT_DIR}")
                shutil.rmtree(REPORT_DIR)
        
        # Ensure directories exist for the new run
        os.makedirs(ROUND_RESULTS_DIR, exist_ok=True)
        os.makedirs(SAVE_PATH, exist_ok=True)
        os.makedirs(SCALER_DIR, exist_ok=True)


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

    def perform_scan(model, df, user_chunks, expected_features, num_clients, baseline_scaler, baseline_mean_per_feature, baseline_std_per_feature):
        results = []
        with torch.no_grad():
            for client_id in range(num_clients):
                # Removed client-specific scaler and stats loading
                # scaler_path = os.path.join(SCALER_DIR, SCALER_FILENAME_TEMPLATE.format(i=client_id))
                # stats_path = os.path.join(SCALER_DIR, f"error_stats_client_{client_id}.pkl")
                # if not os.path.exists(scaler_path): continue
                # with open(scaler_path, "rb") as f:
                #     scaler = pickle.load(f)

                # if os.path.exists(stats_path):
                #     with open(stats_path, "rb") as f:
                #         calib = pickle.load(f)
                #         mean_per_feature, std_per_feature = calib["mean_per_feature"], calib["std_per_feature"]
                # else:
                #     mean_per_feature, std_per_feature = np.full(len(expected_features), 0.02), np.full(len(expected_features), 0.05)

                client_users = user_chunks[client_id]
                for user in client_users:
                    user_df = df[df['user'] == user]
                    u_features = user_df.reindex(columns=expected_features, fill_value=0)
                    u_features = u_features.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
                    u_features = np.log1p(u_features.clip(lower=0))
                    
                    # Use the baseline scaler for all users
                    u_scaled = baseline_scaler.transform(u_features)
                    user_tensor = torch.tensor(u_scaled, dtype=torch.float32).to(DEVICE)

                    user_window_metrics = []
                    
                    if len(user_tensor) >= task.WINDOW_SIZE:
                        windows_to_process = []
                        original_indices = [] # To map back results to original window positions
                        
                        for i in range(0, len(user_tensor) - task.WINDOW_SIZE + 1, SCAN_STRIDE):
                            windows_to_process.append(user_tensor[i : i + task.WINDOW_SIZE])
                            original_indices.append(i)

                        if not windows_to_process:
                            final_score = 0.0
                        else:
                            # Process in batches
                            all_window_errors = []
                            for i in range(0, len(windows_to_process), INFERENCE_BATCH_SIZE):
                                batch_windows = torch.stack(windows_to_process[i : i + INFERENCE_BATCH_SIZE]).to(DEVICE)
                                reconstruction_batch = model(batch_windows)
                                
                                # Calculate squared error for the batch
                                sq_err_batch = torch.mean((reconstruction_batch - batch_windows)**2, dim=1).cpu().numpy()
                                
                                # Apply Z-score and feature selection for each window in the batch
                                for sq_err in sq_err_batch:
                                    # Use the baseline mean and std for Z-score calculation
                                    feat_z = (sq_err - baseline_mean_per_feature) / (baseline_std_per_feature + 1e-6)
                                    pos_feat_z = np.maximum(feat_z, 0)
                                    top_k_z = np.sort(pos_feat_z)[-TOP_K_FEATURES:]
                                    diversity_factor = 1.0 + (np.sum(pos_feat_z > DIVERSITY_THRESHOLD) / len(expected_features))
                                    all_window_errors.append(np.mean(top_k_z) * diversity_factor)
                            
                            user_window_metrics = all_window_errors

                            if user_window_metrics:
                                arr = np.sort(np.array(user_window_metrics))
                                final_score = np.mean(arr[-min(len(arr), PERSISTENCE_WINDOW):])
                            else:
                                final_score = 0.0
                    else: 
                        final_score = 0.0

                    has_insider = (user_df['insider'] != 0).any()
                    results.append({"user": user, "max_z_score": final_score, "is_actual_insider": 1.0 if has_insider else 0.0})
        return pd.DataFrame(results)

    def run_evaluation():
        os.makedirs(ROUND_RESULTS_DIR, exist_ok=True)
        print(f"📊 Starting Evaluation for '{experiment_name}'...")
        
        df = pd.read_csv(DATA_PATH, low_memory=False)
        num_clients = config.get_pyproject("tool", "flwr", "federations", "local-simulation", "options", "num-supernodes") or 10
        user_chunks = np.array_split(sorted(df['user'].unique()), num_clients)

        # --- Baseline Scaler and Error Stats Loading ---
        baseline_scaler = None
        baseline_mean_per_feature = None
        baseline_std_per_feature = None

        baseline_scaler_path = os.path.join(SCALER_DIR, SCALER_FILENAME_TEMPLATE.format(i=0))
        baseline_stats_path = os.path.join(SCALER_DIR, f"error_stats_client_0.pkl")

        if not os.path.exists(baseline_scaler_path):
            print(f"❌ Error: Baseline Scaler file not found at {baseline_scaler_path}. Cannot perform calibrated evaluation.")
            return
        
        with open(baseline_scaler_path, "rb") as f:
            baseline_scaler = pickle.load(f)
            expected_features = list(baseline_scaler.feature_names_in_) # Get expected features from baseline scaler

        if os.path.exists(baseline_stats_path):
            with open(baseline_stats_path, "rb") as f:
                calib = pickle.load(f)
                baseline_mean_per_feature = calib["mean_per_feature"]
                baseline_std_per_feature = calib["std_per_feature"]
        else:
            print(f"⚠️ Warning: Baseline error stats not found at {baseline_stats_path}. Using default values.")
            baseline_mean_per_feature = np.full(len(expected_features), 0.02)
            baseline_std_per_feature = np.full(len(expected_features), 0.05)
        # --- End Baseline Loading ---

        model = task.LSTMAutoencoder(input_dim=len(expected_features), hidden_dim=HIDDEN_DIM).to(DEVICE)
        candidates = []
        if os.path.exists(SAVE_PATH):
            files = [f for f in os.listdir(SAVE_PATH) if f.startswith("parameters_round_") and f.endswith(".pkl")]
            for f in files:
                r_num = int(f.split('_')[-1].split('.')[0])
                if r_num % 2 == 0:
                    candidates.append((r_num, os.path.join(SAVE_PATH, f)))
        candidates = sorted(candidates, key=lambda x: x[0])
        
        summary_data = []
        best_pr_auc, best_round, best_results = 0, -1, None
        comm_log_path = os.path.join(REPORT_DIR, f"communication_log.csv")
        
        total_comm_mb = 0
        if os.path.exists(comm_log_path):
            comm_df = pd.read_csv(comm_log_path, header=None)
            total_comm_mb = comm_df[1].sum()
            avg_comm_per_round = comm_df[1].mean()
        else:
            avg_comm_per_round = 0.0

        for r_num, r_path in candidates:
            print(f"🔄 Evaluating Round {r_num}...")
            with open(r_path, "rb") as f:
                data = pickle.load(f)
                weights = data.get('global_parameters')
                state_dict_keys = model.state_dict().keys()
                model.load_state_dict({k: torch.tensor(w) for k, w in zip(state_dict_keys, weights)})
            model.eval()
            
            current_results = perform_scan(model, df, user_chunks, expected_features, num_clients, 
                                           baseline_scaler, baseline_mean_per_feature, baseline_std_per_feature)
            y_true, y_scores = current_results['is_actual_insider'], current_results['max_z_score']
            pr_auc_val = average_precision_score(y_true, y_scores)
            precision_vals, recall_vals, thresholds_vals = precision_recall_curve(y_true, y_scores)
            
            # Robust F1 score calculation to avoid RuntimeWarning for division by zero
            numerator = 2 * precision_vals * recall_vals
            denominator = precision_vals + recall_vals
            f1_scores = np.zeros_like(numerator)
            non_zero_denominator_mask = denominator > 0
            f1_scores[non_zero_denominator_mask] = numerator[non_zero_denominator_mask] / denominator[non_zero_denominator_mask]

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
            
            convergence_round = -1
            for r, auc in zip(summary_df['Round'], summary_df['PR-AUC']):
                if auc >= 0.95 * best_pr_auc:
                    convergence_round = int(r)
                    break

            best_row = summary_df.loc[summary_df['PR-AUC'].idxmax()]
            experiment_summary = {
                "experiment_name": experiment_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config": config._active_config,
                "best_metrics": best_row.to_dict(),
                "total_communication_mb": total_comm_mb,
                "avg_communication_per_round_mb": avg_comm_per_round,
                "convergence_round": convergence_round,
                "best_round": int(best_round),
                "done": True # Mark as done upon successful completion
            }
            with open(summary_file_path, "w") as f:
                json.dump(experiment_summary, f, indent=4)
            generate_plots(best_results, best_round, pr_auc_val, summary_df, experiment_name, REPORT_DIR)
            return True # Indicate success
        return False # Indicate failure or no results

    def generate_plots(df, best_round, pr_auc_val, summary_df, exp_id, report_path):
        plt.figure(figsize=(10, 6))
        plt.plot(summary_df['Round'], summary_df['PR-AUC'], marker='o', label='PR-AUC')
        plt.plot(summary_df['Round'], summary_df['Balanced_Accuracy'], marker='^', label='Balanced Acc')
        plt.plot(summary_df['Round'], summary_df['Max-F1'], marker='s', label='F1-Score')
        plt.title(f'FL Progress: {exp_id}')
        plt.xlabel('Round Number'); plt.ylabel('Score'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(report_path, 'learning_progress.png'))
        plt.close()

    # --- EXECUTION ---
    experiment_successful = False
    try:
        if mode in ["full", "train"]:
            cleanup_old_metrics()
            
            num_supernodes = config.get_pyproject("tool", "flwr", "federations", "local-simulation", "options", "num-supernodes") or 10
            print(f"🚀 Starting Federated Training for {experiment_name}...")
            subprocess.run(["flower-simulation", "--app", ".", "--num-supernodes", str(num_supernodes)], check=True)
        
        if mode in ["full", "eval"]:
            experiment_successful = run_evaluation()
    except Exception as e:
        print(f"❌ Experiment {experiment_name} failed during execution: {e}")
        import traceback
        traceback.print_exc()
        experiment_successful = False
    
    return experiment_successful


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Experiment Runner")
    parser.add_argument("--mode", choices=["full", "train", "eval"], default="full")
    parser.add_argument("--experiment", type=str, help="Run a specific experiment, or 'all' to run everything", default="all")
    args = parser.parse_args()

    # If this is a child process running a single experiment, execute it
    if args.experiment != "all":
        print(f"\n{'='*60}\n🌟 EXECUTING EXPERIMENT: {args.experiment} (Subprocess)\n{'='*60}")
        run_single_experiment(args.experiment, args.mode)
        print(f"✅ Finished experiment: {args.experiment} (Subprocess)\n")
    else: # This is the main process orchestrating multiple experiments
        from config_manager import config
        experiments_to_run = config.experiment_names
        
        processes = []
        for exp_name in experiments_to_run:
            # Check if experiment is already done in the main process to avoid spawning unnecessary subprocesses
            report_dir = os.path.join(BASE_REPORT_DIR, exp_name)
            summary_file_path = os.path.join(report_dir, "experiment_summary.json")
            if os.path.exists(summary_file_path):
                try:
                    with open(summary_file_path, "r") as f:
                        summary_data = json.load(f)
                    if summary_data.get("done", False) and args.mode != "train":
                        print(f"✅ Experiment '{exp_name}' already completed. Skipping subprocess creation.")
                        continue
                except json.JSONDecodeError:
                    print(f"⚠️ Corrupted experiment_summary.json for '{exp_name}'. Will re-run in subprocess.")

            print(f"\n{'='*60}\n🚀 Spawning subprocess for experiment: {exp_name}\n{'='*60}")
            # Use sys.executable to ensure the correct Python interpreter is used
            cmd = [
                os.sys.executable, # Path to the current Python interpreter
                __file__,          # Path to this script
                "--experiment", exp_name,
                "--mode", args.mode
            ]
            process = subprocess.Popen(cmd)
            processes.append(process)
            process.wait()

        print("\n" + "="*60)
        print("🏁 All experiments finished. Generating final comparison...")
        from compare_experiments import compare_experiments
        compare_experiments()