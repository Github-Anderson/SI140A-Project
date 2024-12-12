import csv
import time
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")

model_1_data = []
with open('100_Overall.csv',newline='') as f:
	reader = csv.reader(f)
	for row in reader:
		model_1_data.append([float(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4])])
model_1_np = np.array(model_1_data)

# Load the example tips dataset
model_1_pd = pd.read_csv('100_Overall.csv', names=['first','second','third','fourth','fifth'])

# Draw a nested boxplot to show bills by day and sex
i1 = sns.boxplot(data=model_1_pd)

sns.despine(offset=10, trim=True,)
i1.get_figure().savefig("100_Overall.png")