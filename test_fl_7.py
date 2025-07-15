import numpy as np

def check_federated_data(npz_path):
    data = np.load(npz_path)
    print("Keys:", list(data.keys()))
    X = data['X']
    y = data['y']

    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")

    # Kiểm tra NaN
    print(f"NaNs in X: {np.isnan(X).sum()}")
    print(f"NaNs in y: {np.isnan(y).sum()}")

    # Nếu y có nhiều cột, phân tích từng cột
    if y.ndim == 2 and y.shape[1] >= 2:
        print(f"Unique labels in first column: {np.unique(y[:,0])}")
        print(f"Unique client ids in second column: {np.unique(y[:,1])}")
    else:
        print(f"Unique labels: {np.unique(y)}")

if __name__ == "__main__":
    check_federated_data('data_pamap2_federated.npz')

