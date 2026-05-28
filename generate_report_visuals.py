import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path("C:/Users/mspcacc/Desktop/college/tez/proj")
# Corrected path for the benchmarks CSV to match compare_experiments.py output
BENCHMARKS_CSV_PATH = PROJECT_ROOT / "experiment_comparison_summary.csv"
RESULTS_BASE_DIR = PROJECT_ROOT / "federated_evaluation_reports"
CENTRALIZED_REPORT_DIR = PROJECT_ROOT / "centralized_evaluation_reports" # New: Centralized report dir
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
        "Type", # Include Type column
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
        "Type": "Type",
        "PR-AUC": "PR-AUC",
        "Max-F1": "Max-F1",
        "Balanced_Acc": "Bal. Acc.",
        "Precision": "Precision",
        "Recall": "Recall",
        "Total_Comm_MB": "Total Comm. (MB)",
        "Conv_Round": "Conv. Round"
    }, inplace=True)

    # --- FIX: Handle NaN/None for centralized model (e.g., Total Comm. (MB), Conv. Round) ---
    # Convert to numeric first, coercing errors to NaN
    df_report["Total Comm. (MB)"] = pd.to_numeric(df_report["Total Comm. (MB)"], errors='coerce')
    df_report["Conv. Round"] = pd.to_numeric(df_report["Conv. Round"], errors='coerce')

    # Now round numeric values
    df_report["Total Comm. (MB)"] = df_report["Total Comm. (MB)"].round(1)

    # Fill NaNs with "N/A". This will convert the column dtype to 'object' if "N/A" is introduced.
    df_report["Total Comm. (MB)"] = df_report["Total Comm. (MB)"].fillna("N/A")
    df_report["Conv. Round"] = df_report["Conv. Round"].fillna("N/A")


    # Format float columns for better presentation (e.g., 3 decimal places)
    for col in ["PR-AUC", "Max-F1", "Bal. Acc.", "Precision", "Recall"]:
        df_report[col] = df_report[col].round(3)
    
    col_widths_dict = {
        "Exp. Config.": 0.20, # Adjusted width
        "Type": 0.08, # New column
        "Total Comm. (MB)": 0.15
    }
    
    create_table_png(df_report, "Table 2: Main Comp. Table", output_dir / "Table_2_Comparison.png", col_widths_dict)


# --- 2. Generate Figure 2 (Learning Curve Graph) ---
def generate_learning_curve_graph(results_base_dir, output_dir):
    all_pr_auc_data = []
    
    # Iterate through each federated experiment subdirectory
    for exp_dir in results_base_dir.iterdir():
        if exp_dir.is_dir():
            exp_name = exp_dir.name
            filepath = exp_dir / "federated_rounds_comparison.csv"

            if filepath.exists():
                try:
                    df_round_metrics = pd.read_csv(filepath)
                    if "Round" in df_round_metrics.columns and "PR-AUC" in df_round_metrics.columns:
                        for index, row in df_round_metrics.iterrows():
                            current_round = int(row["Round"])
                            all_pr_auc_data.append({
                                "Exp. Config.": exp_name,
                                "Round": current_round,
                                "PR-AUC Score": float(row["PR-AUC"])
                            })
                    else:
                        print(f"Warning: 'Round' or 'PR-AUC' column not found in {filepath}. Skipping.")
                except pd.errors.EmptyDataError:
                    print(f"Warning: {filepath} is empty. Skipping.")
                except Exception as e:
                    print(f"Warning: Could not read or parse {filepath}: {e}. Skipping.")
            else:
                print(f"Warning: federated_rounds_comparison.csv not found in {exp_dir}. Skipping.")
    
    # Removed Centralized Model data from here as requested
    
    if not all_pr_auc_data:
        print(f"Warning: No round-by-round metrics found. Cannot generate learning curve graph.")
        return

    df_pr_auc = pd.DataFrame(all_pr_auc_data)

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df_pr_auc, x="Round", y="PR-AUC Score", hue="Exp. Config.", marker="o")
    plt.title("Figure 2: PR-AUC Score Over Rounds (Federated Experiments)", fontsize=16) # Updated title
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("PR-AUC Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Exp.", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    fig_path = output_dir / "Figure_2_Learning_Curve_PR_AUC.png"
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

    # Use the full dataframe, no longer filtering out centralized
    df_plot = df_benchmarks.copy()

    # Ensure 'Total_Comm_MB' is numeric for plotting
    df_plot["Total_Comm_MB"] = pd.to_numeric(df_plot["Total_Comm_MB"], errors='coerce')
    
    # For Centralized model, set Total_Comm_MB to 0 for plotting
    df_plot.loc[df_plot['Experiment'] == 'Centralized', 'Total_Comm_MB'] = 0

    # Drop rows where 'Total_Comm_MB' or 'PR-AUC' might be NaN after coercion (e.g., if 'N/A' was present and couldn't be converted)
    df_plot.dropna(subset=["Total_Comm_MB", "PR-AUC"], inplace=True)

    if df_plot.empty:
        print("Warning: No data with valid communication cost and PR-AUC found for trade-off graph. Skipping.")
        return

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_plot, # Use df_plot
        x="Total_Comm_MB",
        y="PR-AUC",
        hue="Experiment",
        s=100,
        alpha=0.8
    )
    plt.title("Figure 3: Comm. Cost vs. PR-AUC Trade-off (Federated vs. Centralized Experiments)", fontsize=14) # Updated title
    plt.xlabel("Total Comm. Cost (MB)", fontsize=12)
    plt.ylabel("PR-AUC Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Exp.", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    fig_path = output_dir / "Figure_3_Tradeoff_PR_AUC.png"
    plt.savefig(fig_path)
    print(f"Generated Figure 3: {fig_path}")
    plt.close()

# --- 4. Generate Configuration Settings Table ---
def generate_config_table(results_base_dir, centralized_report_dir, output_dir): # Added centralized_report_dir
    config_data = []
    
    # Load Federated Configurations
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

                    plugins_str = ", ".join(efficiency_config.get("active_plugins", [])) or "None"
                    top_k_ratio_val = efficiency_config.get("top_k_ratio", "N/A")

                    # Set Top-K Ratio to "N/A" if top_k plugin is not enabled
                    if "top_k" not in plugins_str and "top_k, quantization" not in plugins_str:
                        top_k_ratio_val = "N/A"

                    config_entry = {
                        "Experiment": exp_summary.get("experiment_name", exp_dir.name),
                        "Type": "Federated",
                        "Non-IID": data_config.get("is_non_iid", "N/A"),
                        "Rounds": federation_config.get("num_rounds", "N/A"),
                        "Local Epochs": federation_config.get("local_epochs", "N/A"),
                        "Frac. Fit": federation_config.get("fraction_fit", "N/A"),
                        "Plugins": plugins_str,
                        "Top-K Ratio": top_k_ratio_val,
                        "LR": model_config.get("learning_rate", "N/A"),
                        "Win. Size": model_config.get("window_size", "N/A"), # Shortened
                        "Hid. Dim": model_config.get("hidden_dim", "N/A") # Shortened
                    }
                    config_data.append(config_entry)
                except Exception as e:
                    print(f"Warning: Could not read or parse {summary_path}: {e}. Skipping.")
            else:
                print(f"Warning: experiment_summary.json not found in {exp_dir}. Skipping.")
    
    # Load Centralized Configuration
    centralized_summary_path = centralized_report_dir / "centralized_experiment_summary.json"
    centralized_config_entry = None
    if centralized_summary_path.exists():
        try:
            with open(centralized_summary_path, 'r') as f:
                centralized_summary = json.load(f)
            
            # For centralized, Plugins and Top-K Ratio are always N/A
            centralized_config_entry = {
                "Experiment": "Centralized",
                "Type": "Centralized",
                "Non-IID": centralized_summary.get("data_config", {}).get("is_non_iid", "N/A"), # Assuming data_config might be nested
                "Rounds": "N/A", # Not applicable
                "Local Epochs": centralized_summary.get("epochs_trained", "N/A"), 
                "Frac. Fit": "N/A", # Not applicable
                "Plugins": "N/A", # Not applicable
                "Top-K Ratio": "N/A", # Always N/A for centralized
                "LR": centralized_summary.get("learning_rate", "N/A"),
                "Win. Size": centralized_summary.get("window_size", "N/A"), # Shortened
                "Hid. Dim": centralized_summary.get("hidden_dim", "N/A") # Shortened
            }
            # Add centralized entry to config_data only if successfully loaded
            # It will be reordered later
            config_data.append(centralized_config_entry)
        except Exception as e:
            print(f"Warning: Could not read or parse {centralized_summary_path}: {e}. Skipping centralized config.")
    else:
        print(f"Warning: Centralized summary file '{centralized_summary_path}' not found. Skipping centralized config.")


    if not config_data:
        print(f"Warning: No configuration data found. Cannot generate config table.")
        return

    df_config = pd.DataFrame(config_data)

    # Reorder to place 'Centralized' at the top
    if centralized_config_entry: # Check if centralized entry was successfully created
        centralized_df = df_config[df_config["Experiment"] == "Centralized"]
        other_configs_df = df_config[df_config["Experiment"] != "Centralized"]
        df_config = pd.concat([centralized_df, other_configs_df], ignore_index=True)


    col_widths_dict = {
        "Experiment": 0.15,
        "Type": 0.08,
        "Non-IID": 0.07,
        "Rounds": 0.07,
        "Local Epochs": 0.09,
        "Frac. Fit": 0.07,
        "Plugins": 0.12,
        "Top-K Ratio": 0.08,
        "LR": 0.07,
        "Win. Size": 0.08, # Updated key
        "Hid. Dim": 0.07 # Updated key
    }

    create_table_png(df_config, "Table 3: Configuration Settings", output_dir / "Table_3_Config_Settings.png", col_widths_dict)


# --- Run all generations ---
print("Generating visuals...")
generate_comparison_table(BENCHMARKS_CSV_PATH, OUTPUT_DIR)
generate_learning_curve_graph(RESULTS_BASE_DIR, OUTPUT_DIR)
generate_tradeoff_graph(BENCHMARKS_CSV_PATH, OUTPUT_DIR)
generate_config_table(RESULTS_BASE_DIR, CENTRALIZED_REPORT_DIR, OUTPUT_DIR) # Updated call with centralized_report_dir
print("Visual generation complete.")