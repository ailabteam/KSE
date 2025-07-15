import numpy as np

data = np.load('data_pamap2_federated.npz', allow_pickle=True)
print(data.files)

for k in data.files:
    print(f"{k}: {type(data[k])}, shape={data[k].shape if hasattr(data[k], 'shape') else 'N/A'}")

