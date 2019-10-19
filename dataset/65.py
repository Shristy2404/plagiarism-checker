import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math

#"project - part D - testing data set.csv" has been renamed as "testingdataset.csv"
#"project - part D - training data set.csv" has been renamed as "trainingdataset.csv"


df_train = pd.read_csv("trainingdataset.csv")
x_train = df_train['Father'].values
x_train = x_train.reshape(-1,1)
y_train = df_train['Son'].values.reshape(-1,1)
print (x_train.shape)
print (y_train.shape)

df_test = pd.read_csv("testingdataset.csv")
x_test = df_test['Father'].values
x_test = x_test.reshape(-1,1)
y_test = df_test['Son'].values.reshape(-1,1)
print (x_test.shape)
print (y_test.shape)

poly = PolynomialFeatures(degree=10)#polynomial degree 10
modified_x_train = poly.fit_transform(x_train)
modified_x_test = poly.fit_transform(x_test)

reg = Lasso()
reg.fit(modified_x_train,y_train)
print("RMSE for training data using Lasso:", math.sqrt(mean_squared_error(y_train,reg.predict(modified_x_train))))
print("RMSE for testing data using Lasso:", math.sqrt(mean_squared_error(y_test,reg.predict(modified_x_test))))
