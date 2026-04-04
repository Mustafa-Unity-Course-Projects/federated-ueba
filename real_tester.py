import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import glob
import re
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
import federated_ueba.task as task
from config_manager import config

# --- DYNAMIC CONFIGURATION ---
DATA_PATH = config.get("data", "processed_data_path")
SCALER_DIR = config.get("data", "scaler_dir")
SCALER_FILENAME_TEMPLATE = config.get("data", "scaler_filename_template")
MODEL_DIR = config.get("federation", "save_path")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use task definition for consistency
WINDOW_SIZE = task.WINDOW_SIZE
STRIDE = task.STRIDE

# The ACTUAL ground-truth malicious users from CERT r4.2
KNOWN_BAD_ACTORS = [
    "AAM0658", "AJR0932", "BDV0168", "BIH0745", "BLS0678",
    "BTL0226", "CAH0936", "DCH0843", "EHB0824", "EHD0584",
    "RGG0064", "TAP0551", "MSO0222"
]

# --- 1. DATA EXTRACTION ENGINE ---
def extract_user_sequences(df, scaler, user_id, expected_features):
    """Extracts all rolling windows for a specific user using the current project's features"""
    user_df = df[df['user'] == user_id]
    if user_df.empty:
        return None

    # Reindex and clean features just like in task.py/insider_detection.py
    u_features = user_df.reindex(columns=expected_features, fill_value=0)
    u_features = u_features.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    
    if len(u_features) < WINDOW_SIZE:
        return None

    scaled_data = scaler.transform(u_features)

    sequences = []
    for i in range(0, len(scaled_data) - WINDOW_SIZE + 1, STRIDE):
        seq = scaled_data[i: i + WINDOW_SIZE]
        sequences.append(seq)

    return torch.tensor(np.array(sequences), dtype=torch.float32).to(DEVICE)


# --- 2. EXECUTION ---
def run_real_data_test():
    print("📥 Loading Project Configuration and Data...")
    
    # 1. Load Data
    if DATA_PATH.endswith(".pkl"):
        df = pd.read_pickle(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # Standardize column types
    df['user'] = df['user'].astype(str).str.strip()
    
    # 2. Identify Features and Scaler
    # We use the first client's scaler as a template for features
    first_scaler_path = os.path.join(SCALER_DIR, SCALER_FILENAME_TEMPLATE.format(i=0))
    if not os.path.exists(first_scaler_path):
        print(f"❌ Error: Missing scaler at {first_scaler_path}. Run training first.")
        return

    with open(first_scaler_path, "rb") as f:
        template_scaler = pickle.load(f)
    expected_features = list(template_scaler.feature_names_in_)
    input_dim = len(expected_features)

    # 3. Setup User Groups
    present_bad_actors = [u for u in KNOWN_BAD_ACTORS if u in df['user'].values]
    if not present_bad_actors:
        print("❌ None of the known bad actors were found in your processed data.")
        # Fallback to searching for users with insider label if available
        if 'insider' in df.columns:
            present_bad_actors = df[df['insider'] != 0]['user'].unique().tolist()
            if present_bad_actors:
                print(f"🕵️ Found {len(present_bad_actors)} users with 'insider' label instead.")

    if not present_bad_actors:
        print("❌ No malicious users identified.")
        return

    print(f"🕵️ Analyzing {len(present_bad_actors)} malicious users.")
    normal_users = [u for u in df['user'].unique() if u not in KNOWN_BAD_ACTORS][:10]

    # 4. Find Model Checkpoints
    pattern = os.path.join(MODEL_DIR, "parameters_round_*.pkl")
    model_files = sorted(glob.glob(pattern), key=lambda x: int(re.search(r'round_(\d+)', x).group(1)))

    if not model_files:
        print(f"❌ No model checkpoints found in {MODEL_DIR}")
        return

    print("\n" + "=" * 90)
    print(f"{'Round':<6} | {'Avg Normal Peak MSE':<20} | {'Avg Bad Actor Peak MSE':<25} | {'Detection Ratio'}")
    print("=" * 90)

    # 5. Load Scaler Mapping
    # Since we use individual scalers, we map each user to their specific scaler
    num_clients = config.get_pyproject("tool", "flwr", "federations", "local-simulation", "options", "num-supernodes")
    unique_users_all = sorted(df['user'].unique())
    user_chunks = np.array_split(unique_users_all, num_clients)
    
    user_to_scaler = {}
    for client_id, users in enumerate(user_chunks):
        s_path = os.path.join(SCALER_DIR, SCALER_FILENAME_TEMPLATE.format(i=client_id))
        if os.path.exists(s_path):
            with open(s_path, "rb") as f:
                s_obj = pickle.load(f)
            for u in users:
                user_to_scaler[u] = s_obj

    # 6. Iterative Evaluation
    for f_path in model_files:
        round_match = re.search(r'round_(\d+)', f_path)
        round_num = int(round_match.group(1))

        # Load Model from project task definition
        model = task.LSTMAutoencoder(input_dim=input_dim, hidden_dim=128).to(DEVICE)
        with open(f_path, "rb") as f:
            data = pickle.load(f)
            # Support both 'global_parameters' and 'globa_parameters' typos
            ndarrays = data.get('global_parameters') or data.get('globa_parameters')
        
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), ndarrays)})
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            def get_max_mse_for_user(user_id):
                scaler = user_to_scaler.get(user_id)
                if scaler is None: return 0.0
                
                tensors = extract_user_sequences(df, scaler, user_id, expected_features)
                if tensors is None: return 0.0
                
                recon = model(tensors)
                # Calculate MSE (Samples, Window, Features) -> (Samples)
                errors = torch.mean((recon - tensors) ** 2, dim=(1, 2)).cpu().numpy()
                return np.max(errors)

            # Analyze Normal Users
            normal_mses = [get_max_mse_for_user(u) for u in normal_users]
            valid_normal = [m for m in normal_mses if m > 0]
            avg_normal_peak = np.mean(valid_normal) if valid_normal else 0.0

            # Analyze Bad Actors
            bad_mses = [get_max_mse_for_user(u) for u in present_bad_actors]
            valid_bad = [m for m in bad_mses if m > 0]
            avg_bad_peak = np.mean(valid_bad) if valid_bad else 0.0

            # Ratio
            ratio = avg_bad_peak / avg_normal_peak if avg_normal_peak > 0 else 0

        print(f"{round_num:<6} | {avg_normal_peak:<20.6f} | {avg_bad_peak:<25.6f} | {ratio:.2f}x")

if __name__ == "__main__":
    run_real_data_test()
