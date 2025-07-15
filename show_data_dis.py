import numpy as np

def display_client_label_distribution(npz_path):
    data = np.load(npz_path)
    X = data['X']
    y = data['y']
    client_ids = data['client_ids']

    print(f"Loaded X shape: {X.shape}, dtype: {X.dtype}")
    print(f"Loaded y shape: {y.shape}, dtype: {y.dtype}")
    print(f"Loaded client_ids shape: {client_ids.shape}, dtype: {client_ids.dtype}")

    unique_clients = np.unique(client_ids)
    unique_labels = np.unique(y)

    print(f"Total samples: {len(y)}")
    print(f"Number of clients: {len(unique_clients)}")
    print(f"Number of unique labels: {len(unique_labels)}\n")

    print("== Label distribution per client ==\n")

    for client in unique_clients:
        client_mask = (client_ids == client)
        client_y = y[client_mask]
        total_client_samples = len(client_y)

        print(f"Client {client} - {total_client_samples} samples - {len(np.unique(client_y))} labels")

        for label in unique_labels:
            label_count = np.sum(client_y == label)
            if label_count > 0:
                label_percent = (label_count / total_client_samples) * 100
                print(f"  Label {label}: {label_count} samples ({label_percent:.2f}%)")
        print()

if __name__ == "__main__":
    npz_file = "data_pamap2_federated_top8_filtered.npz"  # đổi thành file npz của bạn
    display_client_label_distribution(npz_file)

