import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/pokemon.csv')
#data.info()
data.corr()
f,ax = plt.subplots(figsize= (18,18))
sns.heatmap(data.corr(),annot= True, linewidths = .5,fmt='.1f',ax=ax)
#plt.show()
print(data.head(10))
#data.columns


data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
