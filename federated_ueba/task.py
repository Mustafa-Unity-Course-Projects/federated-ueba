import os
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config_manager import config

LEARNING_RATE = config.get("model", "learning_rate")
WINDOW_SIZE = config.get("model", "window_size")
STRIDE = config.get("model", "stride")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATA CONFIGURATION ---
SCALER_DIR = config.get("data", "scaler_dir")
SCALER_FILENAME_TEMPLATE = config.get("data", "scaler_filename_template")
SELECTED_FEATURES = config.get("data", "selected_features") or None

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder: Bidirectional for richer context
        self.encoder = nn.LSTM(input_dim, hidden_dim // 2, num_layers=2, 
                              batch_first=True, dropout=0.2, bidirectional=True)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Decoder: Unidirectional to reconstruct temporal flow
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * 0.01

        # Encoder outputs (B, T, hidden_dim) because it's bidirectional (hidden_dim//2 * 2)
        _, (hidden, _) = self.encoder(x)
        
        # Concat last layers of forward and backward passes
        # hidden shape: (num_layers * num_directions, batch, hidden_size)
        latent = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Pass through bottleneck
        latent = self.bottleneck(latent)
        
        # Prepare for decoder
        decoded_init = latent.unsqueeze(1).repeat(1, WINDOW_SIZE, 1)
        
        x, _ = self.decoder(decoded_init)
        return self.output_layer(x)

def load_partitioned_data(input_path, partition_id, num_partitions):
    df = pd.read_csv(input_path)
    metadata = ['user', 'day', 'week', 'pc', 'activity', 'id', 'label', 'insider', 'to', 'from', 'starttime', 'endtime', 'pcid', 'time_stamp', 'actid']
    features = [c for c in SELECTED_FEATURES if c in df.columns] if SELECTED_FEATURES else [c for c in df.columns if c not in metadata]

    if 'insider' in df.columns:
        df = df[df['insider'] == 0].copy()

    unique_users = sorted(df['user'].unique())
    user_chunks = np.array_split(unique_users, num_partitions)
    my_users = user_chunks[partition_id]
    client_df = df[df['user'].isin(my_users)].copy()

    client_df[features] = client_df[features].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    client_df[features] = np.log1p(client_df[features].clip(lower=0))
    scaler = StandardScaler()
    if not client_df[features].empty:
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
    return torch.utils.data.DataLoader(torch.tensor(np.array(sequences), dtype=torch.float32), batch_size=32, shuffle=True), len(features)

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
            optimizer.step()

def test(net, testloader):
    net.to(DEVICE)
    criterion = nn.MSELoss()
    loss = 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            batch = batch.to(DEVICE)
            loss += criterion(net(batch), batch).item()
    return loss / len(testloader) if len(testloader) > 0 else 0
