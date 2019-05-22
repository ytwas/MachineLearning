import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from subprocess import check_output

data = pd.read_csv('../input/column_2C-weka.csv')
data1 = data[data['class'] = 'Abnormal']
x = np.array(data1.loc[:, 'pelvic_incidence']).reshape(-1,1)
y = np.array(data1.loc[:, 'sacral_slope']).reshape(-1,1)

plt.figure(figsize = [10,10])
plt.scatter(x=x,y=y)
plt.slabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
predict_stapce = np.linspace(min(x), max(x)).reshape(-1,1)
reg.fit(x,y)
predicted = reg.predict(predict_space)
print('R^2 score:', reg.score(x,y))

plt.plot(predict_space,predicted,color='black',linewidth=3)
plt.scatter(x=x, y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()