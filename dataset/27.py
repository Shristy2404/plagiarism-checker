# importing required libraries

import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import math

# Reading train and test data from csv files
train_df = pd.read_csv('project - part D - training data set.csv')
test_df = pd.read_csv('project - part D - testing data set.csv')

# Ignoring first column and reading other columns into X_train and Y_train
X_train = train_df['Father'].values.reshape(-1, 1)
Y_train = train_df['Son'].values.reshape(-1, 1)

# Ignoring first column and reading other columns into X_test and Y_test
X_test = test_df['Father'].values.reshape(-1, 1)
Y_test = test_df['Son'].values.reshape(-1, 1)

poly_degree_ten = PolynomialFeatures(degree=10)
modified_X_train = poly_degree_ten.fit_transform(X_train)
modified_X_test = poly_degree_ten.fit_transform(X_test)

# Creating polynomial regression model of degree 10
print("------ Creating Polynomial Regression Model of degree 10 ------")
poly_reg = LinearRegression()
poly_reg.fit(modified_X_train, Y_train)

train_rmse_poly = math.sqrt(mean_squared_error(Y_train, poly_reg.predict(modified_X_train)))
test_rmse_poly = math.sqrt((mean_squared_error(Y_test, poly_reg.predict(modified_X_test))))

print("Training Error for Polynomial Regression Model of degree 10 is ", train_rmse_poly)
print("Testing Error for Polynomial Regression Model of degree 10 is ", test_rmse_poly)

print("\n")

# implementing lasso regularization on polynomial of degree 10
print("------ Creating Lasso Implementation model  on Polynomial of degree 10 ------")
# lasso_reg = Lasso(max_iter=10000000, tol=0.0000000001) have tried increasing max_iter and decreasing tolerance tol
# But still the model didn't converge hence using default values despite the warning
lasso_reg = Lasso()
lasso_reg.fit(modified_X_train, Y_train)

print("Default Regularization parameter used is ", lasso_reg.alpha)
print("Default max_iter used is ", lasso_reg.max_iter)
print("Default tol used is ", lasso_reg.tol)

train_rmse_lasso = math.sqrt(mean_squared_error(Y_train, lasso_reg.predict(modified_X_train)))
test_rmse_lasso = math.sqrt((mean_squared_error(Y_test, lasso_reg.predict(modified_X_test))))

print("Training Error for Lasso Implementation on Polynomial Regression Model of degree 10 is ", train_rmse_lasso)
print("Testing Error for Lasso Implementation on Polynomial Regression Model of degree 10 is ", test_rmse_lasso)

print("\n")

print("----- Discussing the improvements brought down by lasso "
      "regression in RMSE as compared to the typical regression "
      "model built using the polynomial of degree 10 -----")
if test_rmse_lasso < test_rmse_poly:
    print("Lasso Regularization made testing error lesser")
    if train_rmse_lasso > train_rmse_poly:
        print("Lasso Regularization increased training error")
    else:
        print("Lasso Regularization made training error also smaller")
else:
    print("Lasso Regularization made testing error larger")
    if train_rmse_lasso > train_rmse_poly:
        print("Lasso Regularization increased training error")
    else:
        print("Lasso Regularization made training error also smaller")
