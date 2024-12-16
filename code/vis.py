import torch
import matplotlib.pyplot as plt
import numpy as np

data = np.load('code/data/generated_data_4D.npy')
print(data.shape)
data = data[:, :3]
# Data preprocessing
dataset = torch.Tensor(data).float()  # shape: (10000, 3)

# Visualize data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='red', s=10, edgecolors='black')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.show()