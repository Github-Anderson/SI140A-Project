import csv
import time
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")

model_10_data = []
with open('output.csv',newline='') as f:
	reader = csv.reader(f)
	for row in reader:
		model_10_data.append([float(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4])])
model_10_np = np.array(model_10_data)

# Load the example tips dataset
model_10_pd = pd.read_csv('output.csv', names=['first','second','third','fourth','fifth'])

# Draw a nested boxplot to show bills by day and sex
i1 = sns.boxplot(data=model_10_pd,)

sns.despine(offset=10, trim=True,)
i1.get_figure().savefig("Stimulation.png")