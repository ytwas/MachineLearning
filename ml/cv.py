from sklearn.model_selection import cross_val_score
reg = LinearRegression()
k  = 5
cv_result = cross_val_score(reg, x, y , cv = k)
print('CV Scores:', cv_result)
print('CV scores average: ', np.sum(cv_result)/k)
