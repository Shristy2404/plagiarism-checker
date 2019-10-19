#!/usr/bin/env python
# coding: utf-8

from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import math

df_train = pd.read_csv("project - part D - training data set.csv")
x = df_train['Father'].values.reshape(-1,1)
y = df_train['Son'].values.reshape(-1,1)

df_test = pd.read_csv("project - part D - testing data set.csv")
x_test = df_test['Father'].values.reshape(-1,1)
y_test = df_test['Son'].values.reshape(-1,1)

ply = PolynomialFeatures(degree = 10)
modified_x = ply.fit_transform(x)
modified_x_test = ply.fit_transform(x_test)
reg1 = LinearRegression()
reg1.fit(modified_x,y)
print("For Polynomial regression of degree 10:")
print("Train RMSE: ", math.sqrt(mean_squared_error(y,reg1.predict(modified_x))))
print("Test RMSE: ", math.sqrt(mean_squared_error(y_test,reg1.predict(modified_x_test))))

reg = Lasso()
reg.fit(modified_x, y)
print("For lasso with alpha = 1.0: ")
print("Train RMSE: ", math.sqrt(mean_squared_error(y,reg.predict(modified_x))))
print("Test RMSE: ", math.sqrt(mean_squared_error(y_test,reg.predict(modified_x_test))))