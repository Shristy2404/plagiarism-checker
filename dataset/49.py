import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso

from sklearn.metrics import mean_squared_error

dataset_train = pd.read_csv('project - part D - training data set.csv')
X_train = dataset_train.iloc[::, 1].values.reshape(-1, 1)
y_train = dataset_train.iloc[::, -1]

dataset_test = pd.read_csv('project - part D - testing data set.csv')
X_test = dataset_test.iloc[::, 1].values.reshape(-1, 1)
y_test = dataset_test.iloc[::, -1]

poly = PolynomialFeatures(degree=10)
modified_X_train = poly.fit_transform(X_train, y_train)
modified_X_test = poly.fit_transform(X_test, y_test)

def_reg = LinearRegression()
def_reg.fit(modified_X_train, y_train)
y_def_pred = def_reg.predict(modified_X_test)
mse_def = mean_squared_error(y_test, y_def_pred)
rmse_def = math.sqrt(mse_def)

model = Lasso()
model.fit(modified_X_train, y_train)

y_train_pred = model.predict(modified_X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = math.sqrt(mse_train)

y_test_pred = model.predict(modified_X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = math.sqrt(mse_test)

print('Linear Regression model of degree 10 :')
print('====================================')
print('RMSE --> ', rmse_def)
print('*************************************\n')
print('Lasso Regression model of degree 10 :')
print('===================================')
print('RMSE train --> ', rmse_train)
print('RMSE test --> ', rmse_test)
print('\nRMSE with only Linear Regression for degree 10 is : {0}.\n\tBut RMSE with Lasso Regression is : {1} for the default Alpha value of 1'.format(rmse_def, rmse_test))
