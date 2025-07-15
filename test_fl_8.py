import numpy as np
data = np.load('data_pamap2_federated.npz')
print(data.files)
print("X shape:", data['X'].shape)
print("y shape:", data['y'].shape)
print("Unique client IDs:", np.unique(data['y'][:,1]))
print("Unique labels:", np.unique(data['y'][:,0]))
print("NaNs in X:", np.isnan(data['X']).sum())
print("NaNs in y:", np.isnan(data['y']).sum())
