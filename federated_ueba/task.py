import os
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config_manager import config

# --- CONFIGURATION ---
LEARNING_RATE = config.get("model", "learning_rate") or 0.001
WINDOW_SIZE = config.get("model", "window_size") or 14
STRIDE = config.get("model", "stride") or 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = config.get("data", "batch_size") or 64
SELECTED_FEATURES = config.get("data", "selected_features") or None
HIDDEN_DIM = config.get("model", "hidden_dim") or 64
SCALER_DIR = config.get("data", "scaler_dir")
SCALER_FILENAME_TEMPLATE = config.get("data", "scaler_filename_template")

_cached_df = None

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim // 2, num_layers=2, 
                              batch_first=True, dropout=0.2, bidirectional=True)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * 0.01
        _, (hidden, _) = self.encoder(x)
        latent = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        latent = self.bottleneck(latent)
        decoded_init = latent.unsqueeze(1).repeat(1, WINDOW_SIZE, 1)
        x, _ = self.decoder(decoded_init)
        return self.output_layer(x)

def load_partitioned_data(input_path, partition_id, num_partitions):
    global _cached_df
    if _cached_df is None:
        _cached_df = pd.read_csv(input_path)
    df = _cached_df
    metadata = ['user', 'day', 'week', 'pc', 'activity', 'id', 'label', 'insider', 'to', 'from', 'starttime', 'endtime', 'pcid', 'time_stamp', 'actid']
    features = [c for c in SELECTED_FEATURES if c in df.columns] if SELECTED_FEATURES else [c for c in df.columns if c not in metadata]
    all_users = sorted(df['user'].unique())
    user_chunks = np.array_split(all_users, num_partitions)
    my_users = user_chunks[partition_id]
    client_full_df = df[df['user'].isin(my_users)].copy()
    client_df = client_full_df[client_full_df['insider'] == 0].copy() if 'insider' in client_full_df.columns else client_full_df.copy()
    if client_df.empty: return None, len(features)
    client_df[features] = client_df[features].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    client_df[features] = np.log1p(client_df[features].clip(lower=0))
    scaler = StandardScaler()
    client_df[features] = scaler.fit_transform(client_df[features])
    os.makedirs(SCALER_DIR, exist_ok=True)
    with open(os.path.join(SCALER_DIR, SCALER_FILENAME_TEMPLATE.format(i=partition_id)), "wb") as f:
        pickle.dump(scaler, f)
    sequences = []
    for _, group in client_df.groupby('user'):
        user_data = group[features].values
        if len(user_data) >= WINDOW_SIZE:
            for i in range(0, len(user_data) - WINDOW_SIZE + 1, STRIDE):
                sequences.append(user_data[i:i + WINDOW_SIZE])
    if not sequences: return None, len(features)
    return torch.utils.data.DataLoader(torch.tensor(np.array(sequences), dtype=torch.float32), 
                                       batch_size=BATCH_SIZE, shuffle=True, pin_memory=True), len(features)

def get_error_distribution(net, trainloader):
    """Calculates per-feature error stats for local calibration."""
    net.eval()
    all_errors = []
    with torch.no_grad():
        for batch in trainloader:
            batch = batch.to(DEVICE)
            output = net(batch)
            abs_err = torch.mean(torch.abs(output - batch), dim=1) # (B, D)
            all_errors.append(abs_err.cpu().numpy())
    all_errors = np.concatenate(all_errors, axis=0)
    return np.mean(all_errors, axis=0), np.std(all_errors, axis=0)

def train(net, trainloader, epochs):
    net.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(batch), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

def test(net, testloader):
    net.to(DEVICE)
    criterion = nn.MSELoss()
    loss = 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            batch = batch.to(DEVICE)
            output = net(batch)
            loss += criterion(output, batch).item()
    return loss / len(testloader) if len(testloader) > 0 else 0
