import os
import pickle

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Increased hidden_dim to 128 to handle the 500+ features of CERT 4.2
WINDOW_SIZE = 10
STRIDE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    df = pd.read_csv(input_path)

    # 1. DYNAMIC FEATURE SELECTION (Excluding the "Answers")
    # We remove 'insider', 'label', and 'activity' to prevent data leakage
    metadata = ['user', 'day', 'pc', 'activity', 'id', 'label', 'insider', 'to', 'from']
    features = [c for c in df.columns if c not in metadata]

    # 2. Partitioning
    unique_users = sorted(df['user'].unique())
    user_chunks = np.array_split(unique_users, num_partitions)
    my_users = user_chunks[partition_id]
    client_df = df[df['user'].isin(my_users)].copy()

    # 3. Numeric Conversion & Cleanup
    # Forcing all 500+ features to float32 and filling NaNs
    client_df[features] = client_df[features].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

    # 4. Scaling (Prevents the 10^15 loss explosion)
    scaler = MinMaxScaler()
    if not client_df[features].empty:
        client_df[features] = scaler.fit_transform(client_df[features])
        # Save scaler for later anomaly detection testing
        scaler_path = f"./scaler_data/scaler_client_{partition_id}.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"💾 Scaler saved: {scaler_path}")

    # 5. Sequence Building
    sequences = []
    client_df = client_df.sort_values(['user', 'day'])

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
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in trainloader:
            # Batch shape is (Batch, Window, Features)
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            reconstruction = net(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}: Train Loss {running_loss / len(trainloader):.6f}")


def test(net, testloader):
    """
    This is the function the error was complaining about.
    It calculates the Reconstruction Error (MSE).
    """
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