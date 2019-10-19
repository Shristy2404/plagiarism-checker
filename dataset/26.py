#!/usr/bin/env python
# coding: utf-8


# In[173]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# In[134]:


training_data_set = pd.read_csv('project - part D - training data set.csv')
testing_data_set = pd.read_csv('project - part D - testing data set.csv')
x_train = training_data_set['Father'].values.reshape(-1,1)
y_train = training_data_set['Son'].values.reshape(-1,1)
x_test = testing_data_set['Father'].values.reshape(-1,1)
y_test = testing_data_set['Son'].values.reshape(-1,1)


# In[135]:


def obtain_polynomial_x(x, degree_required):
    poly = PolynomialFeatures(degree=degree_required, include_bias=False)
    modified_x = poly.fit_transform(x)
    return modified_x



# In[136]:


def lasso_reg_runner(max_degree, x_train, y_train, x_test, y_test, train_err, test_err ):
    result=[]
    for i in range(1, max_degree+1):
        modified_x_train = obtain_polynomial_x(x_train, i)
        modified_x_test = obtain_polynomial_x(x_test, i)
        reg = Lasso().fit(modified_x_train, y_train)
        y_train_predict = reg.predict(modified_x_train)
        train_data_mse = mean_squared_error(y_train_predict, y_train)
        train_data_rmse = math.sqrt(train_data_mse)
        y_test_predict = reg.predict(modified_x_test)
        test_data_mse = mean_squared_error(y_test_predict, y_test)
        test_data_rmse = math.sqrt(test_data_mse)
        train_err.append(train_data_rmse)
        test_err.append(test_data_rmse)
        result.append({'degree': i,
                       'coefficients': reg.coef_,
                       'intercept': reg.intercept_,
                       'trainRMSE': train_data_rmse,
                       'testRMSE': test_data_rmse})
    return result;



# In[138]:


train_err_lasso=[]
test_err_lasso=[]
lasso_result = lasso_reg_runner(10, x_train, y_train, x_test, y_test, train_err_lasso, test_err_lasso)


# In[174]:


print('The training RMSE for Degree 10 after Lasso Regularization is : '+ str(lasso_result[9]['trainRMSE']))
print('The testing RMSE for Degree 10 after Lasso Regularization is : '+ str(lasso_result[9]['testRMSE']))


# In[ ]:




