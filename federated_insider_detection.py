import os
import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve, auc
import federated_ueba.task as task
from config_manager import config

# --- CONFIGURATION ---
SAVE_PATH = config.get("federation", "save_path")
DATA_PATH = config.get("data", "processed_data_path")
SCALER_DIR = config.get("data", "scaler_dir")
SCALER_FILENAME_TEMPLATE = config.get("data", "scaler_filename_template")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPORT_DIR = "federated_evaluation_reports"
ROUND_RESULTS_DIR = os.path.join(REPORT_DIR, "round_by_round_results")

HIDDEN_DIM = config.get("model", "hidden_dim") or 64
TOP_K_FEATURES = 5
PERSISTENCE_WINDOW = 3 
DIVERSITY_THRESHOLD = 2.0 
SCAN_STRIDE = 1

def get_candidate_rounds(step=10):
    if not os.path.exists(SAVE_PATH): return []
    files = [f for f in os.listdir(SAVE_PATH) if f.startswith("parameters_round_") and f.endswith(".pkl")]
    rounds = []
    for f in files:
        r_num = int(f.split('_')[-1].split('.')[0])
        if r_num % step == 0:
            rounds.append((r_num, os.path.join(SAVE_PATH, f)))
    return sorted(rounds, key=lambda x: x[0])

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
                        abs_err = torch.mean(torch.abs(reconstruction - window), dim=1).squeeze(0).cpu().numpy()
                        feat_z = (abs_err - mean_per_feature) / (std_per_feature + 1e-6)
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

def run_federated_search():
    print(f"🚀 Starting Multi-Round Model Search (every 10th round)...")
    os.makedirs(ROUND_RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.sort_values(['user', 'day' if 'day' in df.columns else 'week'])
    num_clients = config.get_pyproject("tool", "flwr", "federations", "local-simulation", "options", "num-supernodes") or 10
    user_chunks = np.array_split(sorted(df['user'].unique()), num_clients)

    with open(os.path.join(SCALER_DIR, SCALER_FILENAME_TEMPLATE.format(i=0)), "rb") as f:
        expected_features = list(pickle.load(f).feature_names_in_)

    model = task.LSTMAutoencoder(input_dim=len(expected_features), hidden_dim=HIDDEN_DIM).to(DEVICE)
    candidates = get_candidate_rounds(step=10)
    
    summary_data = []
    best_pr_auc = 0
    best_round = -1
    best_results = None

    for r_num, r_path in candidates:
        print(f"🔄 Evaluating Round {r_num}...")
        with open(r_path, "rb") as f:
            data = pickle.load(f)
            weights = data.get('global_parameters') or data.get('globa_parameters')
            state_dict_keys = model.state_dict().keys()
            model.load_state_dict({k: torch.tensor(w) for k, w in zip(state_dict_keys, weights)})
        model.eval()
        
        current_results = perform_scan(model, df, user_chunks, expected_features, num_clients)
        
        y_true, y_scores = current_results['is_actual_insider'], current_results['max_z_score']
        
        # PR-AUC (Average Precision) is much more robust for imbalanced datasets
        pr_auc_val = average_precision_score(y_true, y_scores)
        
        # Calculate max F1 and its corresponding threshold
        precision_vals, recall_vals, thresholds_vals = precision_recall_curve(y_true, y_scores)
        f1_scores = np.where((precision_vals + recall_vals) > 0, (2 * precision_vals * recall_vals) / (precision_vals + recall_vals), 0)
        best_f1_idx = np.argmax(f1_scores)
        
        summary_data.append({
            "Round": r_num,
            "PR-AUC": pr_auc_val,
            "Max-F1": f1_scores[best_f1_idx],
            "Precision": precision_vals[best_f1_idx],
            "Recall": recall_vals[best_f1_idx],
            "Optimal-Threshold": thresholds_vals[best_f1_idx] if best_f1_idx < len(thresholds_vals) else thresholds_vals[-1]
        })
        
        # Save individual round results
        current_results.to_csv(os.path.join(ROUND_RESULTS_DIR, f"round_{r_num}_results.csv"), index=False)
        print(f"  Round {r_num} PR-AUC: {pr_auc_val:.4f} | Max F1: {f1_scores[best_f1_idx]:.4f}")

        if pr_auc_val > best_pr_auc:
            best_pr_auc, best_round, best_results = pr_auc_val, r_num, current_results

    # Save overall summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(REPORT_DIR, "federated_rounds_comparison.csv"), index=False)
    
    print(f"\n🏆 Best Model Found: Round {best_round} (PR-AUC: {best_pr_auc:.4f})")
    best_results.to_csv("federated_insider_results_advanced.csv", index=False)
    return best_results, best_round, best_pr_auc, summary_df

def generate_report(df, best_round, pr_auc_val, summary_df):
    print("\n📊 Generating Comparison Reports focused on Precision-Recall...")
    
    # Plot 1: Learning Curve (PR-AUC and F1 over rounds)
    plt.figure(figsize=(10, 6))
    plt.plot(summary_df['Round'], summary_df['PR-AUC'], marker='o', label='PR-AUC (Avg Precision)', color='blue')
    plt.plot(summary_df['Round'], summary_df['Max-F1'], marker='s', label='Max F1-Score', color='green')
    plt.title('Federated Learning Progress: Precision-Recall Focus')
    plt.xlabel('Round Number')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(REPORT_DIR, 'learning_progress_pr.png'))

    # Plot 2: Final PR Curve for Best Round
    y_true, y_scores = df['is_actual_insider'], df['max_z_score']
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR Curve (AUC: {pr_auc_val:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (Best Round {best_round})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(REPORT_DIR, 'best_pr_curve.png'))

    # Histogram of scores
    plt.figure(figsize=(10, 6))
    plt.hist(df[df['is_actual_insider'] == 0]['max_z_score'], bins=100, alpha=0.5, label='Normal', color='gray', log=True)
    plt.hist(df[df['is_actual_insider'] == 1]['max_z_score'], bins=100, alpha=0.8, label='Insider', color='red', log=True)
    plt.title(f'Score Distribution: Round {best_round} | PR-AUC: {pr_auc_val:.4f}')
    plt.xlabel('Detection Score')
    plt.ylabel('User Count (Log Scale)')
    plt.legend()
    plt.savefig(os.path.join(REPORT_DIR, 'best_round_score_dist.png'))
    
    print("\n" + "=" * 55)
    print(f"       FEDERATED PR-EVALUATION COMPLETE")
    print("=" * 55)
    print(f"Winner Round:             {best_round}")
    print(f"PR-AUC (Avg Precision):   {pr_auc_val:.4f}")
    print(f"Max F1-Score:             {summary_df[summary_df['Round']==best_round]['Max-F1'].values[0]:.4f}")
    print(f"Comparison CSV: {os.path.join(REPORT_DIR, 'federated_rounds_comparison.csv')}")
    print("=" * 55)

if __name__ == "__main__":
    results, r_num, pr_auc, summary = run_federated_search()
    generate_report(results, r_num, pr_auc, summary)
