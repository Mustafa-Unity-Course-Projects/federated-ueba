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

# --- CONFIGURATION ---
DATA_PATH = config.get("data", "processed_data_path")
CENTRALIZED_MODEL_PATH = "centralized_model.pth"
CENTRALIZED_SCALER_PATH = "centralized_scaler.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = config.get("model", "learning_rate") or 0.001
SELECTED_FEATURES = config.get("data", "selected_features") or None
EARLY_STOPPING_PATIENCE = 10 
HIDDEN_DIM = config.get("model", "hidden_dim") or 64
TOP_K_FEATURES = 5

def train_centralized():
    print(f"🚀 Starting Bidirectional Centralized Model Training on {DEVICE}...")

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)

    # 2. Dynamic Feature Selection
    metadata = ['user', 'day', 'week', 'pc', 'activity', 'id', 'label', 'insider', 'to', 'from', 'starttime', 'endtime', 'pcid', 'time_stamp', 'actid']
    features = [c for c in SELECTED_FEATURES if c in df.columns] if SELECTED_FEATURES else [c for c in df.columns if c not in metadata]

    print(f"📊 Training with {len(features)} features.")

    # 3. Filter Training Data (Normal only)
    if 'insider' in df.columns:
        df = df[df['insider'] == 0].copy()
    
    # 4. Numeric Conversion & Scaling
    df[features] = df[features].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    df[features] = np.log1p(df[features].clip(lower=0))
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    with open(CENTRALIZED_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # 5. Sequence Building
    unique_users = df['user'].unique()
    train_users, val_users = train_test_split(unique_users, test_size=0.2, random_state=42)

    def get_sequences(user_list):
        seqs = []
        for user in user_list:
            user_data = df[df['user'] == user].sort_values('day')[features].values
            if len(user_data) >= task.WINDOW_SIZE:
                for i in range(0, len(user_data) - task.WINDOW_SIZE + 1, task.STRIDE):
                    seqs.append(user_data[i:i + task.WINDOW_SIZE])
        return seqs

    train_seqs = get_sequences(train_users)
    val_seqs = get_sequences(val_users)
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
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Val Loss: {avg_val_loss:.6f}")
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                break

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), CENTRALIZED_MODEL_PATH)
    
    # 7. Advanced Distribution Stats
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
    with open("centralized_error_stats.pkl", "wb") as f:
        pickle.dump(error_stats, f)
    
    print(f"💾 Optimized Stats saved.")

if __name__ == "__main__":
    train_centralized()
