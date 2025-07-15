import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

data = np.load("data_pamap2.npz")
X = data['X']

print("Before:", X.shape)
selector = VarianceThreshold(threshold=1e-6)
X = selector.fit_transform(X)
print("After variance threshold:", X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("NaNs after scaling:", np.isnan(X_scaled).sum())

