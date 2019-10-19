#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import math
from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

dataset_train = pd.read_csv('project - part D - training data set.csv')
dataset_test = pd.read_csv('project - part D - testing data set.csv')

x_train = dataset_train.pop('Father').values.reshape(-1,1)
y_train = dataset_train.pop('Son').values.reshape(-1,1)

x_test = dataset_test.pop('Father').values.reshape(-1,1)
y_test = dataset_test.pop('Son').values.reshape(-1,1)

#print('x_train = ', x_train.shape)
#print('y_train = ', y_train.shape)
#print('x_test = ', x_test.shape)
#print('y_test = ', y_test.shape)
#print()

poly = PolynomialFeatures(degree=10)
x_train_modified = poly.fit_transform(x_train)
x_test_modified = poly.fit_transform(x_test)

#print('x_train_modified:', x_train_modified.shape)
#print('x_test_modified:', x_test_modified.shape)
#print()

reg = LinearRegression()
res = reg.fit(x_train_modified, y_train)

print('Predicted Coefficient Value = ', reg.coef_)
print('Predicted Intercept Value = ', reg.intercept_)
print()

y_pred_train = reg.predict(x_train_modified)
lMSE_train = metrics.mean_squared_error(y_train, y_pred_train)
lRMSE_train = math.sqrt(lMSE_train)
lR2_train = reg.score(x_train_modified, y_train)

y_pred_test = reg.predict(x_test_modified)
lMSE_test = metrics.mean_squared_error(y_test, y_pred_test)
lRMSE_test = math.sqrt(lMSE_test)
lR2_test = reg.score(x_test_modified, y_test)

print('Training RMSE: ', lRMSE_train)
#print('Training R2: ', lR2_train)
print('Testing RMSE: ', lRMSE_test)
#print('Testing R2: ', lR2_test)
print()

reg_Lasso = Lasso(max_iter=1000000,tol=1e-3)
#print(reg_Lasso.alpha)
reg_Lasso = reg_Lasso.fit(x_train_modified, y_train)

print('Predicted Lasso Coefficient Value = ', reg_Lasso.coef_)
print('Predicted Lasso Intercept Value = ', reg_Lasso.intercept_)
print('Iteration: ', reg_Lasso.n_iter_)
print()

y_pred_train_l = reg_Lasso.predict(x_train_modified)
lMSE_train_l = metrics.mean_squared_error(y_train, y_pred_train_l)
lRMSE_train_l = math.sqrt(lMSE_train_l)
lR2_train_l = reg_Lasso.score(x_train_modified, y_train)

y_pred_test_l = reg_Lasso.predict(x_test_modified)
lMSE_test_l = metrics.mean_squared_error(y_test, y_pred_test_l)
lRMSE_test_l = math.sqrt(lMSE_test_l)
lR2_test_l = reg_Lasso.score(x_test_modified, y_test)

print('Lasso Training RMSE: ', lRMSE_train_l)
#print('Lasso Training R2: ', lR2_train_l)
print('Lasso Testing RMSE: ', lRMSE_test_l)
#print('Lasso Testing R2: ', lR2_test_l)
print()

if(lRMSE_test_l < lRMSE_test):
    print('Lasso Regression helped to improve the model for Polynomial Degree 10 by combatting overfitting')
else:
    print('Lasso Regression didn\'t help to combat overfitting for Polynomial Degree 10')


# In[ ]:





# In[ ]:




