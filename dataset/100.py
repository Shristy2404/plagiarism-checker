#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import math
import warnings
warnings.filterwarnings('ignore')


# In[16]:


print('Reading training data from file project - part D - training data set.csv/ train.csv')
try:
    train_df = pd.read_csv('project - part D - training data set.csv')
except:
    train_df = pd.read_csv('train.csv')
train_x = train_df['Father'].values.reshape(-1,1)
train_y = train_df['Son'].values.reshape(-1,1)
print('Reading training data from file project - part D - testing data set.csv/ test.csv')
try:
    test_df = pd.read_csv('project - part D - testing data set.csv')
except:
    test_df = pd.read_csv('test.csv')
test_x = test_df['Father'].values.reshape(-1,1)
test_y = test_df['Son'].values.reshape(-1,1)


# In[17]:


degree=10
# iter_array=[1000,5000,10000,50000,100000,500000]
# train_err = []
# test_err = []
print("Number of Training points are ", train_x.shape[0])
print("Number of Testing points are ", test_x.shape[0])
print("")
poly = PolynomialFeatures(degree=degree,include_bias=False)
modified_train_x = poly.fit_transform(train_x)
modified_test_x = poly.fit_transform(test_x)
# for max_iter in iter_array:
print("Lasso with default alpha")
print("\n")
print("\n")
reg=Lasso()
reg.fit(modified_train_x,train_y)
print("Lasso Train RMSE is: " , math.sqrt(mean_squared_error(train_y,reg.predict(modified_train_x))) )
print("Lasso Test RMSE is: ", math.sqrt(mean_squared_error(test_y,reg.predict(modified_test_x))))
# train_err.append(math.sqrt(mean_squared_error(train_y,reg.predict(modified_train_x))))
# test_err.append(math.sqrt(mean_squared_error(test_y,reg.predict(modified_test_x))))
# print(reg.coef_)
# plt.xlabel('Iterations')
# plt.ylabel('RMSE')
# plt.plot(iter_array , train_err , 'bo-', label='Training')
# plt.plot(iter_array , test_err , 'ro-' , label='Test')
# plt.legend()
print("\n")
print("Polynomial Regression with degree 10")
print("\n")
print("\n")
reg = LinearRegression()
reg.fit(modified_train_x,train_y)
print("Polynomial Regression degree 10 Train RMSE is: " , math.sqrt(mean_squared_error(train_y,reg.predict(modified_train_x))))
print("Polynomial Regression degree 10  Test RMSE is: ", math.sqrt(mean_squared_error(test_y,reg.predict(modified_test_x))))

print("\n")
print("\n")
print("""Polynomial Regression of degree 10 gives too much of freedom for each coefficient to overfit specific to  training data
That is why training error is very low for Polynomial Regression of degree 10. But this will not be close to underlying generality and 
Thats is why testing error is high in Polynomial Regression of degree 10.""")
print("\n")
print("Improvements brought down by Lasso Regression:")
print("\n")
print("""
But by tuning regularization parameter alpha in Lasso regression we reduce the freedom for coefficients (penalize) 
and that can remove the overfitting in Polynomial Regression of degree 10 which increased the training error and 
decreased the test error and making the fit close to the underlying generality. Alpha needs to be tuned to get the ideal scenario. 
""")


# In[ ]:




