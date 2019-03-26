import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from subprocess import check_output

data = pd.read_csv('../input/column_2C-weka.csv')

data.head()

data.info()

data.describe()

color_list = ['red' if i == 'Abnormal' else 'green' for i in data.loc[:,'class']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
                           c = color_list,
                           figsize = [15,15],
                           diagonal = 'hist',
                           alpha = 0.5,
                           s = 200,
                           marker = '*',
                           edgecolor = "black")

plt.show()


sns.countplot(x = "class", data = data)
data.loc[:, 'class'].value_counts()
