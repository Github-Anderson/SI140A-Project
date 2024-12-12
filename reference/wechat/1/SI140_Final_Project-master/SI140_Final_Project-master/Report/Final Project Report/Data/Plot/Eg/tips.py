# encoding:utf-8
import seaborn as sns
import pandas as pd

tips = pd.read_csv("tips.csv")
print(tips)
sns.jointplot("total_bill", "tip", tips, kind='reg')
sns.plt.show()
sns.plt.close()  # 用于清空之前的图表。若不清空，那么下一个图表会残留上一个图表的点

sns.lmplot("total_bill", "tip", tips, col="smoker")
sns.plt.show()