import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.linalg import pinv
from sklearn.linear_model import Lasso

# Load the MATLAB data
cat_data = loadmat('catData_w.mat')
dog_data = loadmat('dogData_w.mat')

cat_wave = cat_data['cat_wave']
dog_wave = dog_data['dog_wave']

# Combine the data
CD = np.hstack((dog_wave, cat_wave))

# Split into training and testing sets
train = np.hstack((dog_wave[:, :60], cat_wave[:, :60]))
test = np.hstack((dog_wave[:, 60:80], cat_wave[:, 60:80]))

# Create labels
labels = np.concatenate((np.ones(60), -1 * np.ones(60)))

# --- First part of the code ---
A = labels @ pinv(train)
test_labels = np.sign(A @ test)

plt.figure(1)
plt.subplot(4, 1, 1)
plt.bar(np.arange(len(test_labels)), test_labels, color=[0.6, 0.6, 0.6], edgecolor='k')
plt.axis('off')

plt.subplot(4, 1, 2)
plt.bar(np.arange(len(A)), A, color=[0.6, 0.6, 0.6], edgecolor='k')
plt.axis([0, 1024, -0.002, 0.002])
plt.axis('off')

plt.figure(2)
plt.subplot(2, 2, 1)
A2 = np.flipud(A.reshape(32, 32))
plt.pcolormesh(A2, cmap='gray')
plt.axis('off')

# --- Second part of the code (Lasso) ---
lasso = Lasso(alpha=0.1)
lasso.fit(train.T, labels.T)
A_lasso = lasso.coef_.T
test_labels_lasso = np.sign(A_lasso.T @ test)

plt.figure(1)
plt.subplot(4, 1, 3)
plt.bar(np.arange(len(test_labels_lasso)), test_labels_lasso, color=[0.6, 0.6, 0.6], edgecolor='k')
plt.axis('off')

plt.subplot(4, 1, 4)
plt.bar(np.arange(len(A_lasso)), A_lasso, color=[0.6, 0.6, 0.6], edgecolor='k')
plt.axis([0, 1024, -0.008, 0.008])
plt.axis('off')

plt.figure(2)
plt.subplot(2, 2, 2)
A2_lasso = np.flipud(A_lasso.reshape(32, 32))
plt.pcolormesh(A2_lasso, cmap='gray')
plt.axis('off')

plt.show()
