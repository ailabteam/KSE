import numpy as np

def check_nan_in_npz(npz_path):
    data = np.load(npz_path)
    X = data['X']
    y = data['y']

    nan_in_features = np.isnan(X).sum()
    nan_in_labels = np.isnan(y).sum() if y.dtype.kind == 'f' else 0  # chỉ check NaN cho float labels

    print(f"NaNs in features: {nan_in_features}, NaNs in labels: {nan_in_labels}")

if __name__ == "__main__":
    # Thay 'data_pamap2_federated.npz' bằng đường dẫn file của bạn
    check_nan_in_npz('data_pamap2_federated.npz')

