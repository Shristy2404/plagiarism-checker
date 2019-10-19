#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures

dataset_train = pd.read_csv("project - part D - training data set.csv")
dataset_test = pd.read_csv("project - part D - testing data set.csv")

y_train = dataset_train.pop('Son')
x_train= dataset_train.pop('Father')

y_test = dataset_test.pop('Son')
x_test = dataset_test.pop('Father')
train_err =[]
test_err=[]

poly = PolynomialFeatures(10)
X_poly = poly.fit_transform(x_train.values.reshape(-1,1)) 

poly.fit(X_poly, y_train.values.reshape(-1,1)) 
    
X_poly_test = poly.fit_transform(x_test.values.reshape(-1,1)) 

poly.fit(X_poly_test, y_test.values.reshape(-1,1)) 

 
reg = Lasso(alpha=0.5)
reg.fit(X_poly,y_train.values.reshape(-1,1))
y_pred_train = reg.predict(X_poly)
y_pred_test = reg.predict(X_poly_test)
print("Regression Coefficients: ",reg.coef_)

mse_train = metrics.mean_squared_error(y_train,y_pred_train)

mse_test = metrics.mean_squared_error(y_test,y_pred_test)

print("Train RMSE: ",math.sqrt(mse_train))
print("Test RMSE: ",math.sqrt(mse_test))
