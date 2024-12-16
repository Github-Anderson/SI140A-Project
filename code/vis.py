import torch
import matplotlib.pyplot as plt
import numpy as np

data = np.load(r'C:\Users\23714\Desktop\Shanghai Tech\大二上合集\概率论\project\generated_data_4D.npy')
# data = np.load(r'C:\Users\23714\Desktop\Shanghai Tech\大二上合集\概率论\project\simulated_data_from_raw_2.npy')
print(data.shape)
data = data[:, :3]
# 数据预处理
dataset = torch.Tensor(data).float()  # shape: (10000, 3)

# 可视化数据
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='red')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.show()