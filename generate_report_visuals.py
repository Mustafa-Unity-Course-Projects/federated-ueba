import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path("C:/Users/mspcacc/Desktop/college/tez/proj")
# Corrected path for the benchmarks CSV
BENCHMARKS_CSV_PATH = PROJECT_ROOT / "federated_evaluation_reports" / "federated_experiment_comparison.csv"
RESULTS_BASE_DIR = PROJECT_ROOT / "federated_evaluation_reports"
OUTPUT_DIR = PROJECT_ROOT / "generated_visuals"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper function to create tables ---
def create_table_png(df, title, output_path, col_widths_dict=None):
    if df.empty:
        print(f"Warning: No data in the DataFrame for {title}. Cannot generate table PNG.")
        return

    fig, ax = plt.subplots(figsize=(12, len(df) * 0.8 + 1))
    ax.axis('off')
    ax.set_position([0, 0, 1, 1]) # Make axes fill the figure

    num_cols = len(df.columns)
    
    # Calculate column widths
    col_widths = []
    if col_widths_dict:
        fixed_width_sum = sum(col_widths_dict.values())
        remaining_width = 1.0 - fixed_width_sum
        num_other_cols = num_cols - len(col_widths_dict)
        default_width = remaining_width / num_other_cols if num_other_cols > 0 else 0

        for col_name in df.columns.tolist():
            col_widths.append(col_widths_dict.get(col_name, default_width))
    else:
        col_widths = [1.0 / num_cols] * num_cols if num_cols > 0 else []

    table = ax.table(cellText=df.values,
                     colLabels=df.columns.tolist(),
                     cellLoc='center',
                     loc='center',
                     colWidths=col_widths)

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Style cells
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        if row == 0: # Header row
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor('#cccccc')
        else:
            cell.set_facecolor('white')

    plt.savefig(output_path, bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close(fig)
    print(f"Generated {title}: {output_path}")


# --- 1. Generate Table 2 (Main Comparison Table) ---
def generate_comparison_table(csv_path, output_dir):
    try:
        df_benchmarks = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Cannot generate comparison table. Please ensure 'compare_experiments.py' has been run successfully.")
        return

    # Select and reorder columns for the report
    report_columns = [
        "Experiment",
        "PR-AUC",
        "Max-F1",
        "Balanced_Acc",
        "Precision",
        "Recall",
        "Total_Comm_MB",
        "Conv_Round"
    ]

    # Check if all expected columns exist
    missing_columns = [col for col in report_columns if col not in df_benchmarks.columns]
    if missing_columns:
        print(f"Error: Missing columns in {csv_path}: {missing_columns}. Please check the CSV content.")
        return

    df_report = df_benchmarks[report_columns].copy()

    # Rename columns for better readability in the report
    df_report.rename(columns={
        "Experiment": "Exp. Config.",
        "PR-AUC": "PR-AUC",
        "Max-F1": "Max-F1",
        "Balanced_Acc": "Bal. Acc.",
        "Precision": "Precision",
        "Recall": "Recall",
        "Total_Comm_MB": "Total Comm. (MB)",
        "Conv_Round": "Conv. Round"
    }, inplace=True)

    # Format float columns for better presentation (e.g., 3 decimal places)
    for col in ["PR-AUC", "Max-F1", "Bal. Acc.", "Precision", "Recall"]:
        df_report[col] = df_report[col].round(3)
    df_report["Total Comm. (MB)"] = df_report["Total Comm. (MB)"].round(1)
    df_report["Conv. Round"] = df_report["Conv. Round"].astype(int)

    col_widths_dict = {
        "Exp. Config.": 0.25,
        "Total Comm. (MB)": 0.15
    }
    
    create_table_png(df_report, "Table 2: Main Comp. Table", output_dir / "Table_2_Comparison.png", col_widths_dict)


# --- 2. Generate Figure 2 (Learning Curve Graph) ---
def generate_learning_curve_graph(results_base_dir, output_dir):
    all_f1_data = []
    
    # Iterate through each experiment subdirectory
    for exp_dir in results_base_dir.iterdir():
        if exp_dir.is_dir():
            exp_name = exp_dir.name
            # Look for federated_rounds_comparison.csv inside the experiment directory
            filepath = exp_dir / "federated_rounds_comparison.csv"

            if filepath.exists():
                try:
                    df_round_metrics = pd.read_csv(filepath)
                    # Assuming 'Round' and 'Max-F1' columns exist in this CSV
                    if "Round" in df_round_metrics.columns and "Max-F1" in df_round_metrics.columns:
                        for index, row in df_round_metrics.iterrows():
                            all_f1_data.append({
                                "Exp. Config.": exp_name,
                                "Round": int(row["Round"]),
                                "Max-F1 Score": float(row["Max-F1"])
                            })
                    else:
                        print(f"Warning: 'Round' or 'Max-F1' column not found in {filepath}. Skipping.")
                except pd.errors.EmptyDataError:
                    print(f"Warning: {filepath} is empty. Skipping.")
                except Exception as e:
                    print(f"Warning: Could not read or parse {filepath}: {e}. Skipping.")
            else:
                print(f"Warning: federated_rounds_comparison.csv not found in {exp_dir}. Skipping.")
    
    if not all_f1_data:
        print(f"Warning: No round-by-round metrics found in subdirectories of {results_base_dir}. Cannot generate learning curve graph.")
        return

    df_f1 = pd.DataFrame(all_f1_data)

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df_f1, x="Round", y="Max-F1 Score", hue="Exp. Config.", marker="o")
    plt.title("Figure 2: Max F1 Score Over Rounds", fontsize=16)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Max F1 Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Exp.", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    fig_path = output_dir / "Figure_2_Learning_Curve.png"
    plt.savefig(fig_path)
    print(f"Generated Figure 2: {fig_path}")
    plt.close()

# --- 3. Generate Figure 3 (Communication Cost vs. Detection Performance Graph) ---
def generate_tradeoff_graph(csv_path, output_dir):
    try:
        df_benchmarks = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Cannot generate trade-off graph. Please ensure 'compare_experiments.py' has been run successfully.")
        return

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_benchmarks,
        x="Total_Comm_MB",
        y="Max-F1",
        hue="Experiment",
        s=100,
        alpha=0.8
    )
    plt.title("Figure 3: Comm. Cost vs. Det. Perf. Trade-off", fontsize=14)
    plt.xlabel("Total Comm. Cost (MB)", fontsize=12)
    plt.ylabel("Max F1 Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Exp.", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    fig_path = output_dir / "Figure_3_Tradeoff.png"
    plt.savefig(fig_path)
    print(f"Generated Figure 3: {fig_path}")
    plt.close()

# --- 4. Generate Configuration Settings Table ---
def generate_config_table(results_base_dir, output_dir):
    config_data = []
    for exp_dir in results_base_dir.iterdir():
        if exp_dir.is_dir():
            summary_path = exp_dir / "experiment_summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, 'r') as f:
                        exp_summary = json.load(f)
                    
                    config = exp_summary.get("config", {})
                    federation_config = config.get("federation", {})
                    efficiency_config = config.get("efficiency", {})
                    model_config = config.get("model", {})
                    data_config = config.get("data", {})

                    config_entry = {
                        "Experiment": exp_summary.get("experiment_name", exp_dir.name),
                        "Non-IID": data_config.get("is_non_iid", "N/A"),
                        "Rounds": federation_config.get("num_rounds", "N/A"),
                        "Local Epochs": federation_config.get("local_epochs", "N/A"),
                        "Frac. Fit": federation_config.get("fraction_fit", "N/A"),
                        "Plugins": ", ".join(efficiency_config.get("active_plugins", [])),
                        "Top-K Ratio": efficiency_config.get("top_k_ratio", "N/A"),
                        "LR": model_config.get("learning_rate", "N/A"),
                        "Window Size": model_config.get("window_size", "N/A"),
                        "Hidden Dim": model_config.get("hidden_dim", "N/A")
                    }
                    config_data.append(config_entry)
                except Exception as e:
                    print(f"Warning: Could not read or parse {summary_path}: {e}. Skipping.")
            else:
                print(f"Warning: experiment_summary.json not found in {exp_dir}. Skipping.")
    
    if not config_data:
        print(f"Warning: No configuration data found in subdirectories of {results_base_dir}. Cannot generate config table.")
        return

    df_config = pd.DataFrame(config_data)

    col_widths_dict = {
        "Experiment": 0.15,
        "Non-IID": 0.08,
        "Rounds": 0.08,
        "Local Epochs": 0.1,
        "Frac. Fit": 0.08,
        "Plugins": 0.15,
        "Top-K Ratio": 0.1,
        "LR": 0.08,
        "Window Size": 0.1,
        "Hidden Dim": 0.08
    }

    create_table_png(df_config, "Table 3: Configuration Settings", output_dir / "Table_3_Config_Settings.png", col_widths_dict)


# --- Run all generations ---
print("Generating visuals...")
generate_comparison_table(BENCHMARKS_CSV_PATH, OUTPUT_DIR)
generate_learning_curve_graph(RESULTS_BASE_DIR, OUTPUT_DIR)
generate_tradeoff_graph(BENCHMARKS_CSV_PATH, OUTPUT_DIR)
generate_config_table(RESULTS_BASE_DIR, OUTPUT_DIR) # New call
print("Visual generation complete.")