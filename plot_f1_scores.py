import os
import pandas as pd
import matplotlib.pyplot as plt
import json

BASE_REPORT_DIR = "federated_evaluation_reports"

def plot_all_f1_scores():
    """
    Reads federated_rounds_comparison.csv from all experiment directories
    and plots their F1 scores over rounds.
    """
    print(f"📊 Generating F1 score comparison plot from {BASE_REPORT_DIR}...")

    experiment_dirs = [d for d in os.listdir(BASE_REPORT_DIR) if os.path.isdir(os.path.join(BASE_REPORT_DIR, d))]

    if not experiment_dirs:
        print(f"No experiment directories found in {BASE_REPORT_DIR}.")
        return

    plt.figure(figsize=(12, 8))
    
    for exp_name in sorted(experiment_dirs):
        report_dir = os.path.join(BASE_REPORT_DIR, exp_name)
        summary_file_path = os.path.join(report_dir, "experiment_summary.json")
        
        # Check if the experiment was marked as done
        is_done = False
        if os.path.exists(summary_file_path):
            try:
                with open(summary_file_path, "r") as f:
                    summary_data = json.load(f)
                if summary_data.get("done", False):
                    is_done = True
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Corrupted experiment_summary.json for '{exp_name}'.")

        if not is_done:
            print(f"Skipping incomplete experiment: {exp_name}")
            continue

        csv_path = os.path.join(report_dir, "federated_rounds_comparison.csv")

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'Round' in df.columns and 'Max-F1' in df.columns:
                    plt.plot(df['Round'], df['Max-F1'], marker='o', label=exp_name)
                    print(f"Loaded F1 scores for experiment: {exp_name}")
                else:
                    print(f"⚠️ Warning: 'Round' or 'Max-F1' column not found in {csv_path} for experiment {exp_name}.")
            except Exception as e:
                print(f"❌ Error reading {csv_path} for experiment {exp_name}: {e}")
        else:
            print(f"⚠️ Warning: {csv_path} not found for experiment {exp_name}.")

    plt.title('Max F1 Score Over Rounds for All Experiments')
    plt.xlabel('Round Number')
    plt.ylabel('Max F1 Score')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    
    plot_filename = os.path.join(BASE_REPORT_DIR, "all_experiments_f1_comparison.png")
    plt.savefig(plot_filename)
    print(f"✅ F1 score comparison plot saved to {plot_filename}")
    plt.show()

if __name__ == "__main__":
    plot_all_f1_scores()
