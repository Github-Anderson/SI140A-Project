import csv
import time
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

model_3_data = []
with open('3.csv',newline='') as f:
	reader = csv.reader(f)
	for row in reader:
		model_3_data.append([float(row[0]),float(row[1]),float(row[2])])
model_3_np = np.array(model_3_data)


model_3_pd = pd.read_csv('D:/DataScience/Wechat/3.csv', names=['first','second','third'])
model_5_pd = pd.read_csv('D:/DataScience/Wechat/5.csv', names=['first','second','third','fourth','fifth'])
sns.set(style="whitegrid", color_codes=True)
model_3_plot = sns.violinplot(data=model_3_pd)
model_3_plot.get_figure().savefig("sns.png")