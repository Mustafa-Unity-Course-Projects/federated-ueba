import os
import pickle

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config_manager import config

LEARNING_RATE = config.get("model", "learning_rate")
WINDOW_SIZE = config.get("model", "window_size")
STRIDE = config.get("model", "stride")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATA CONFIGURATION ---
SCALER_DIR = config.get("data", "scaler_dir")
SCALER_FILENAME_TEMPLATE = config.get("data", "scaler_filename_template")

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        # Repeat hidden state for each time step
        x = hidden.repeat(WINDOW_SIZE, 1, 1).transpose(0, 1)
        x, _ = self.decoder(x)
        return self.output_layer(x)

def load_partitioned_data(input_path, partition_id, num_partitions):
    with open(input_path, "rb") as f:
        df = pickle.load(f)

    # 1. DYNAMIC FEATURE SELECTION (Excluding the "Answers")
    # We remove metadata and target columns to prevent data leakage
    metadata = ['user', 'day', 'week', 'pc', 'activity', 'id', 'label', 'insider', 'to', 'from', 'starttime', 'endtime', 'pcid', 'time_stamp', 'actid']
    features = [c for c in df.columns if c not in metadata]

    # 2. Partitioning
    unique_users = sorted(df['user'].unique())
    user_chunks = np.array_split(unique_users, num_partitions)
    my_users = user_chunks[partition_id]
    client_df = df[df['user'].isin(my_users)].copy()

    # 3. Numeric Conversion & Cleanup
    client_df[features] = client_df[features].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

    # 4. Scaling
    scaler = MinMaxScaler()
    if not client_df[features].empty:
        client_df[features] = scaler.fit_transform(client_df[features])
        
        # Ensure the directory exists
        os.makedirs(SCALER_DIR, exist_ok=True)
        
        # Save scaler using configuration template
        scaler_filename = SCALER_FILENAME_TEMPLATE.format(i=partition_id)
        scaler_path = os.path.join(SCALER_DIR, scaler_filename)
        
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"💾 Scaler saved: {scaler_path}")

    # 5. Sequence Building
    sequences = []
    # Use 'day' if available, otherwise 'week', otherwise just 'user'
    time_col = 'day' if 'day' in client_df.columns else ('week' if 'week' in client_df.columns else None)
    sort_cols = ['user', time_col] if time_col else ['user']
    client_df = client_df.sort_values(sort_cols)

    for _, group in client_df.groupby('user'):
        user_data = group[features].values
        if len(user_data) >= WINDOW_SIZE:
            for i in range(0, len(user_data) - WINDOW_SIZE + 1, STRIDE):
                sequences.append(user_data[i:i + WINDOW_SIZE])

    if not sequences:
        return None, len(features)

    # Convert to 3D Tensor: (Samples, Time_Steps, Features)
    loader = torch.utils.data.DataLoader(
        torch.tensor(np.array(sequences)),
        batch_size=32,
        shuffle=True
    )

    return loader, len(features)


def train(net, trainloader, epochs):
    net.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    net.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in trainloader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            reconstruction = net(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}: Train Loss {running_loss / len(trainloader):.6f}")


def test(net, testloader):
    net.to(DEVICE)
    criterion = nn.MSELoss()
    loss = 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            batch = batch.to(DEVICE)
            reconstruction = net(batch)
            loss += criterion(reconstruction, batch).item()

    return loss / len(testloader) if len(testloader) > 0 else 0
