import numpy as np
data = np.load('data_pamap2.npz')
y = data['y']
print("Unique labels:", np.unique(y))
print("Label counts:", {k: (y == k).sum() for k in np.unique(y)})

