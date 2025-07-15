import numpy as np

def check_npz_clients(npz_file):
    data = np.load(npz_file)
    print(f"Keys: {list(data.keys())}")
    X = data['X']
    y = data['y']
    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")

    # Client IDs nằm ở cột cuối cùng của y
    client_ids = y[:, -1].astype(int)
    print(f"Unique client IDs: {np.unique(client_ids)}")

    # Labels là cột đầu (hoặc nhiều cột đầu, tùy bạn)
    labels = y[:, 0]
    print(f"Unique labels: {np.unique(labels)}")

    # Check NaN
    print(f"NaNs in X: {np.isnan(X).sum()}")
    print(f"NaNs in y: {np.isnan(y).sum()}")

if __name__ == "__main__":
    check_npz_clients('data_pamap2_federated.npz')

