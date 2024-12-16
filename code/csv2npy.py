import pandas as pd
import numpy as np

# Load the data from a CSV file
df = pd.read_csv("code/csv/data3.csv")

# Convert the data to a numpy array
data = df.to_numpy()

# Save the data as a .npy file
np.save("code/data/raw_data_3.npy", data)
