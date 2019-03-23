import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from subprocess import check_output

data = pd.read_csv('../input/pokemon.csv')

series = data['Defense']
#print(type(series))
data_frame = data[['Defense']]
#print(type(data_frame))
x = data['Defense'] > 200
#print(data[x])

print(data[np.logical_and(data['Defense']>200,data['Attack']>100)])