import numpy as np

data = np.load('data_pamap2_federated.npz')
X = data['X']
y = data['y']

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Giả sử:
# y[:, 0] là label
# y[:, 1] là client_id

labels = y[:, 0].astype(int)
client_ids = y[:, 1].astype(int)

unique_clients = np.unique(client_ids)
print(f"Number of clients: {len(unique_clients)}")

X_clients = []
y_clients = []

for c in unique_clients:
    idx = np.where(client_ids == c)[0]
    X_clients.append(X[idx])
    y_clients.append(labels[idx])

print(f"Client 0 data shape: {X_clients[0].shape}, labels shape: {y_clients[0].shape}")

