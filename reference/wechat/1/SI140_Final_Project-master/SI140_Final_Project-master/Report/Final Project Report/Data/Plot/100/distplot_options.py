"""
Distribution plot options
=========================

"""
import csv
import time
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

model_10_data = []
with open('100_5.csv',newline='') as f:
	reader = csv.reader(f)
	for row in reader:
		model_10_data.append([float(row[0])])
model_10_np = np.array(model_10_data)

sns.set(style="white", palette="muted", color_codes=True)

# Set up the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)

# Generate a random univariate dataset
# d = rs.normal(size=100)

# Plot a simple histogram with binsize determined automatically
i1 = sns.distplot(model_10_np, kde=False, color="b", ax=axes[0, 0])

# Plot a kernel density estimate and rug plot
sns.distplot(model_10_np, hist=False, rug=True, color="r", ax=axes[0, 1])

# Plot a filled kernel density estimate
sns.distplot(model_10_np, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])

# Plot a historgram and kernel density estimate
sns.distplot(model_10_np, color="m", ax=axes[1, 1])

#plt.setp(axes, yticks=[])
plt.tight_layout()
i1.get_figure().savefig("100_5.png")