import numpy as np
import pandas as pandas
import matplotlib.pyplot as plt

# lendo os dados do dataset
dataframe = pandas.read_csv("data/ex1data1.txt", header=None)
dataset = dataframe.values

x = dataset[:, 0:1].astype(float)
y = dataset[:, 1].astype(float)

colors = np.random.rand(2)

plt.scatter(x, y, s=0.5, alpha=0.5)
plt.show()
