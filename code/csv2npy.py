import pandas as pd
import numpy as np

# 步骤1: 读取CSV文件
df = pd.read_csv(r"C:\Users\23714\Desktop\Shanghai Tech\大二上合集\概率论\project\data2.csv")

# 步骤2: 转换为NumPy数组

data = df.to_numpy()


# 步骤3: 保存为.npy文件
np.save(r"C:\Users\23714\Desktop\Shanghai Tech\大二上合集\概率论\project\raw_data_2.npy", data)
