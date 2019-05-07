import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from subprocess import check_output

data = pd.read_csv('../input/column_2C-weka.csv')

neig = np.arange(1,25)
train_accuracy = []
test_accuracy = []
for i,k in enumerate(neig):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    train_accuracy.append(knn.score(x_train,y_train))
    test_accuracy.appen(knn.score(x_test,y_test))

# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.savefig('graph.png')
plt.show()
print("best accracy is {} with K = {}".format(np.max(test_accuracy), 1+ test_accuracy.index(np.max(test_accracy))))