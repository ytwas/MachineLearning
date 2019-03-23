import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from subprocess import check_output

data = pd.read_csv('../input/pokemon.csv')

data.plot(kind = 'scatter', x = 'Attack', y = 'Defense', alpha = 0.5, color = 'red')
plt.xlabel('Attack')
plt.ylabel('Dfence')
plt.title('Attack Dfence Scatter plot')
plt.show()
