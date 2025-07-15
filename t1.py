import numpy as np

data = np.load('data_pamap2_federated.npz')
X = data['X']
y = data['y']
print(X.shape, y.shape)
print(np.unique(y))

