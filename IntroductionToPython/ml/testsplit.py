import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from subprocess import check_output

data = pd.read_csv('../input/column_2C-weka.csv')

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 1)
knn = KNeighboursClassifier(n_neighbours = 3)
x ,y = data.loc[:, data.columns != 'class'],data.loc[:,'class']
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print('With KNN (K=3) accuracy is:', knn.score(x_test, y_test))

