import numpy as np

def check_distribution_per_client(filename):
    data = np.load(filename)
    X = data['X']           # shape (N, window_size, features)
    y = data['y']           # shape (N,) labels (encoded)
    client_ids = data['client_ids']  # shape (N,)
    
    print(f"Loaded X shape: {X.shape}, dtype: {X.dtype}")
    print(f"Loaded y shape: {y.shape}, dtype: {y.dtype}")
    print(f"Loaded client_ids shape: {client_ids.shape}, dtype: {client_ids.dtype}")
    
    print(f"Total samples: {len(y)}")
    print(f"NaNs in X: {np.isnan(X).sum()}")
    print(f"NaNs in y: {np.isnan(y).sum()}")
    
    labels, counts = np.unique(y, return_counts=True)
    print(f"Unique labels (encoded): {labels}")
    print("Global label distribution:")
    for label, count in zip(labels, counts):
        print(f"Label {label}: {count} samples")
    
    clients = np.unique(client_ids)
    print(f"\nNumber of clients: {len(clients)}")
    print(f"Clients: {clients}")
    
    print("\n== Label distribution per client ==")
    for client in clients:
        idxs = np.where(client_ids == client)[0]
        client_labels = y[idxs]
        cl_labels, cl_counts = np.unique(client_labels, return_counts=True)
        print(f"Client {client} - {len(idxs)} samples - {len(cl_labels)} labels")
        for cl_label, cl_count in zip(cl_labels, cl_counts):
            print(f"  Label {cl_label}: {cl_count} samples")

if __name__ == "__main__":
    filename = "data_pamap2_federated_top8.npz"
    check_distribution_per_client(filename)

