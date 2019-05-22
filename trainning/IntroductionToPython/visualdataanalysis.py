import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from subprocess import check_output

data = pd.read_csv('../input/pokemon.csv')
data1 = data.loc[:, ["Attack", "Defense", "Speed"]]
data1.plot()
data1.plot(subplots = True)
plt.show()
#scatter plot
data1.plot(kind = "scatter", x = "Attack", y = "Defense")
plt.show()
#hist plot
data1.plot(kind = "hist" ,y = "Defense", bins = 50, range(0,250),normed = True)


#hist subplots with non cumulative and cumulative
fig, axes = plt.subplots(nrows = 2, ncols = 1)
data1.plot(kind = "hist", y = "Defense", bins = 50, range = (0,250), normed = True, ax = axes[0])
data1.plot(kind = "hist", y = "Defense", bins = 50, range = (0,250), normed = True, ax = axes[1],cumulative = True)
plt.savefig('graph.png')
