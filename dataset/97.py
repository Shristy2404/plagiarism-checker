import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn import metrics
import math
DataFrame1 = pd.read_csv('project - part D - training data set.csv')
DataFrame2 = pd.read_csv('project - part D - testing data set.csv')
#Training Data
X_train = DataFrame1['Father'].values/10000
X_train = X_train.reshape(-1,1)
y_train = DataFrame1['Son'].values.reshape(-1,1)
#testing Data
X_test = DataFrame2['Father'].values/10000
X_test = X_test.reshape(-1,1)
y_test = DataFrame2['Son'].values.reshape(-1,1)
poly = PolynomialFeatures(degree=10)
Modified_X_train = poly.fit_transform(X_train)
Modified_X_test = poly.fit_transform(X_test)
reg = LinearRegression()
reg.fit(Modified_X_train,y_train)
Poly_Reg_Train_RMSE = math.sqrt(metrics.mean_squared_error(y_train,reg.predict(Modified_X_train)))
Poly_Reg_Test_RMSE = math.sqrt(metrics.mean_squared_error(y_test,reg.predict(Modified_X_test)))
les = Lasso ()
les.fit(Modified_X_train,y_train)
Lasso_Reg_Train_RMSE = math.sqrt(metrics.mean_squared_error(y_train,les.predict(Modified_X_train)))
Lasso_Reg_Test_RMSE = math.sqrt(metrics.mean_squared_error(y_test,les.predict(Modified_X_test)))
print('Train RMSE of polynomial regression of model of degree 10 is :',Poly_Reg_Train_RMSE)
print('Test RMSE of polynomial regression of model of degree 10 is :',Poly_Reg_Test_RMSE)
print('Train RMSE of Lasso regression of model of degree 10 is :',Lasso_Reg_Train_RMSE)
print('Test RMSE of Lasso regression of model of degree 10 is :',Lasso_Reg_Test_RMSE)
#Discusion over lasso and normal polynomial regression
#Here we can see the lasso regression model is giving higher Test RMSE,
# I believe because ,it has been suggested in problem to take the regularization as default (which is 1)
#And we know that for higher values of lamda regularization parameter ,we will be having Higher bias, less variance 
#resulting in not optimal model,but worse model than normal polynomial regression as we are getting higher bias impacting the Error
#i.e Expected error = bias^2+variance+noise, so increased bias ,impacting error in lasso
#We should be taking the optimal value of regularization parameter
#i.e not so low(shouldnt be 0 or closer to 0),not so high(shouldnt be1 or closer to 1)