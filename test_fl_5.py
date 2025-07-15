import numpy as np
data = np.load('data_pamap2_federated.npz')
print("Unique clients:", np.unique(data['y'][:, 1]))  # cột 1 là client_id
print("All client ids:", np.unique(data['y'][:, 1]))

