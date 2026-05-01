import os
import json
import pandas as pd

BASE_REPORT_DIR = "federated_evaluation_reports"

def compare_experiments():
    print("📊 Comparing Federated Experiment Results...")
    
    experiment_summaries = []
    
    if not os.path.exists(BASE_REPORT_DIR):
        print(f"Error: Base report directory '{BASE_REPORT_DIR}' not found.")
        return

    # Iterate through each experiment directory
    for exp_name in os.listdir(BASE_REPORT_DIR):
        exp_path = os.path.join(BASE_REPORT_DIR, exp_name)
        if os.path.isdir(exp_path):
            summary_file = os.path.join(exp_path, "experiment_summary.json")
            if os.path.exists(summary_file):
                with open(summary_file, "r") as f:
                    try:
                        summary = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in {summary_file}. Skipping.")
                        continue
                    
                    # Extract relevant data
                    best_metrics = summary.get("best_metrics", {})
                    data = {
                        "Experiment": summary.get("experiment_name", exp_name),
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
    
    if not experiment_summaries:
        print("No experiment summaries found to compare.")
        return

    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame(experiment_summaries)
    
    # Sort by PR-AUC descending
    if "PR-AUC" in comparison_df.columns:
        comparison_df = comparison_df.sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)

    print("\n" + "="*120)
    print("                                 FEDERATED EXPERIMENT COMPARISON")
    print("="*120)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(comparison_df.to_string(index=False))
    print("="*120)
    
    output_csv_path = os.path.join(BASE_REPORT_DIR, "federated_experiment_comparison.csv")
    comparison_df.to_csv(output_csv_path, index=False)
    print(f"\nDetailed comparison saved to '{output_csv_path}'")

if __name__ == "__main__":
    compare_experiments()
