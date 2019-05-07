from sklearn.model_selection import GridSearchCV
grid = {'n_neighbors' : np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv=3)
knn_cv.fit(x,y)

print("Tuned hyperparameter k:{}".format(knn_cv.best_params_))

print("Best score: {}".format(knn_cv.best_score_))