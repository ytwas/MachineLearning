from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
x,y = data.loc[:,data.columns != 'class'],data.loc[:,'class']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, radom_state = 1)
rf = RandomForestClassifier(random_state = 4)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix: \n', cm)
print('Classification report: \n', classification_report(y_test, y_pred))
