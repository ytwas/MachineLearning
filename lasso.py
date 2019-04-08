from sklearn.linear_model import Lasso
x = np.array(data1.loc[:,['pelvic_incidence', 'pelvic_tiltnumeric', 'lumbar_lordosis_angle','pelvic_radius']])
x_train, x_test, y_train, y_test = train_test_split(x,y,randomstate = 3, test_size = 0.3)
lasso = Lasso(alpha = 0.1, normalize = True)
lasso.fit(x_train,y_train)
ridge_predict = lasso.predict(x_test)
print('Lasso score:', lasso.score(x_test,y_test))
print('Lasso coefficients: ',lasso.coef_)
