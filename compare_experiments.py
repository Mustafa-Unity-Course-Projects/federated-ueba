import os
import json
import pandas as pd

BASE_FEDERATED_REPORT_DIR = "federated_evaluation_reports"
CENTRALIZED_REPORT_DIR = "centralized_evaluation_reports"

def compare_experiments():
    print("📊 Comparing Experiment Results (Federated & Centralized)...")

    experiment_summaries = []
    
    # --- 1. Load Federated Experiment Summaries ---
    if os.path.exists(BASE_FEDERATED_REPORT_DIR):
        for exp_name in os.listdir(BASE_FEDERATED_REPORT_DIR):
            exp_path = os.path.join(BASE_FEDERATED_REPORT_DIR, exp_name)
            if os.path.isdir(exp_path):
                summary_file = os.path.join(exp_path, "experiment_summary.json")
                if os.path.exists(summary_file):
                    with open(summary_file, "r") as f:
                        try:
                            summary = json.load(f)
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON in {summary_file}. Skipping.")
                            continue
                        
                        best_metrics = summary.get("best_metrics", {})
                        data = {
                            "Experiment": summary.get("experiment_name", exp_name),
                            "Type": "Federated",
                            "Best_Round": summary.get("best_round"),
                            "PR-AUC": best_metrics.get("PR-AUC"),
                            "Max-F1": best_metrics.get("Max-F1"),
                            "Balanced_Acc": best_metrics.get("Balanced_Accuracy"),
                            "Precision": best_metrics.get("Precision"),
                            "Recall": best_metrics.get("Recall"),
                            "Total_Comm_MB": summary.get("total_communication_mb"),
                            "Conv_Round": summary.get("convergence_round"),
                        }
                        experiment_summaries.append(data)
    else:
        print(f"Warning: Federated report directory '{BASE_FEDERATED_REPORT_DIR}' not found.")

    # --- 2. Load Centralized Experiment Summary ---
    centralized_summary_file = os.path.join(CENTRALIZED_REPORT_DIR, "centralized_experiment_summary.json")
    if os.path.exists(centralized_summary_file):
        with open(centralized_summary_file, "r") as f:
            try:
                centralized_summary = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in {centralized_summary_file}. Skipping centralized results.")
            else:
                ad_metrics = centralized_summary.get("anomaly_detection_metrics", {})
                centralized_data = {
                    "Experiment": "Centralized", # Fixed name for centralized
                    "Type": "Centralized",
                    "Best_Round": None, # Not applicable for centralized
                    "PR-AUC": ad_metrics.get("pr_auc"),
                    "Max-F1": ad_metrics.get("best_f1"),
                    "Balanced_Acc": ad_metrics.get("balanced_accuracy_at_best_f1"),
                    "Precision": ad_metrics.get("precision_at_best_f1"),
                    "Recall": ad_metrics.get("recall_at_best_f1"),
                    "Total_Comm_MB": None, # Not applicable for centralized
                    "Conv_Round": None, # Not applicable for centralized
                }
                experiment_summaries.append(centralized_data)
    else:
        print(f"Warning: Centralized summary file '{centralized_summary_file}' not found. Run train_centralized.py first.")


    if not experiment_summaries:
        print("No experiment summaries found to compare.")
        return

    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame(experiment_summaries)
    
    # Sort by PR-AUC descending
    if "PR-AUC" in comparison_df.columns:
        comparison_df = comparison_df.sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)

    print("\n" + "="*120)
    print("                         FEDERATED vs. CENTRALIZED EXPERIMENT COMPARISON")
    print("="*120)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(comparison_df.to_string(index=False))
    print("="*120)
    
    output_csv_path = os.path.join("experiment_comparison_summary.csv") # Save to project root
    comparison_df.to_csv(output_csv_path, index=False)
    print(f"\nDetailed comparison saved to '{output_csv_path}'")

if __name__ == "__main__":
    compare_experiments()
