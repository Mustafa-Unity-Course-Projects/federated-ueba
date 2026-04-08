import os
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config_manager import config

# --- CONFIGURATION ---
LEARNING_RATE = config.get("model", "learning_rate") or 0.0008
WINDOW_SIZE = config.get("model", "window_size") or 14
STRIDE = config.get("model", "stride") or 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = config.get("data", "batch_size") or 128
SELECTED_FEATURES = config.get("data", "selected_features") or None
HIDDEN_DIM = config.get("model", "hidden_dim") or 128
SCALER_DIR = config.get("data", "scaler_dir")
SCALER_FILENAME_TEMPLATE = config.get("data", "scaler_filename_template")
TEST_SIZE = config.get("data", "test_size") or 0.2

_cached_df = None

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim // 2, num_layers=2, 
                              batch_first=True, dropout=0.2, bidirectional=True)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        if self.training:
            # Denoising: help prevent overfitting to noise/outliers
            x = x + torch.randn_like(x) * 0.02
        
        _, (hidden, _) = self.encoder(x)
        latent = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        latent = self.bottleneck(latent)
        decoded_init = latent.unsqueeze(1).repeat(1, WINDOW_SIZE, 1)
        x_recon, _ = self.decoder(decoded_init)
        return self.output_layer(x_recon)

def load_partitioned_data(input_path, partition_id, num_partitions):
    global _cached_df
    if _cached_df is None:
        _cached_df = pd.read_pickle(input_path) if input_path.endswith('.pkl') else pd.read_csv(input_path)
            
    df = _cached_df
    metadata = ['user', 'day', 'week', 'pc', 'activity', 'id', 'label', 'insider', 'to', 'from', 'starttime', 'endtime', 'pcid', 'time_stamp', 'actid']
    features = [c for c in SELECTED_FEATURES if c in df.columns] if SELECTED_FEATURES else [c for c in df.columns if c not in metadata]
    
    all_users = sorted(df['user'].unique())
    user_chunks = np.array_split(all_users, num_partitions)
    my_users = user_chunks[partition_id]
    
    client_full_df = df[df['user'].isin(my_users)].copy()
    # Model only learns from "Normal" behavior
    client_df = client_full_df[client_full_df['insider'] == 0].copy() if 'insider' in client_full_df.columns else client_full_df.copy()
    
    if client_df.empty: return None, None, len(features)
        
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
                
    if not sequences: return None, None, len(features)
    
    # Split into Train and Validation to monitor overfitting
    train_seq, val_seq = train_test_split(np.array(sequences), test_size=TEST_SIZE, random_state=42)
        
    train_loader = torch.utils.data.DataLoader(torch.tensor(train_seq, dtype=torch.float32), 
                                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.tensor(val_seq, dtype=torch.float32), 
                                            batch_size=BATCH_SIZE, shuffle=False)
                                            
    return train_loader, val_loader, len(features)

def train(net, trainloader, valloader, epochs):
    net.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for batch in trainloader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(batch), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation step
        val_loss = test(net, valloader)
        print(f"  Epoch {epoch+1}: Train Loss: {train_loss/len(trainloader):.6f} | Val Loss: {val_loss:.6f}")

def test(net, testloader):
    if testloader is None: return 0.0
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

def get_error_distribution(net, trainloader):
    net.eval()
    all_errors = []
    with torch.no_grad():
        for batch in trainloader:
            batch = batch.to(DEVICE)
            output = net(batch)
            sq_err = torch.mean((output - batch)**2, dim=1)
            all_errors.append(sq_err.cpu().numpy())
    if not all_errors: return np.zeros(1), np.ones(1)
    all_errors = np.concatenate(all_errors, axis=0)
    return np.mean(all_errors, axis=0), np.std(all_errors, axis=0)
