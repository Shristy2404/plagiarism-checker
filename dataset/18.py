#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# Remember what this line did?
get_ipython().run_line_magic('matplotlib', 'inline')
import math 

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


# In[3]:


dataset_train = pd.read_csv('project - part D - training data set.csv')
X_Train = dataset_train['Father'].values.reshape(-1,1)
y_Train = dataset_train['Son'].values.reshape(-1,1)

dataset_test = pd.read_csv('project - part D - testing data set.csv')
X_Test = dataset_test['Father'].values.reshape(-1,1)
y_Test = dataset_test['Son'].values.reshape(-1,1)


# In[32]:


poly = PolynomialFeatures(degree = 10)
modified_X_Train = poly.fit_transform(X_Train)
modified_X_Test = poly.fit_transform(X_Test)


regressor_linear = LinearRegression()
regressor_linear.fit(modified_X_Train, y_Train)
Train_RMSE_linear = get_error(y_Train, regressor_linear.predict(modified_X_Train))
Test_RMSE_linear = get_error(y_Test, regressor_linear.predict(modified_X_Test))
print("Train RMSE in linear: ", Train_RMSE_linear)
print("Test RMSE in linear: ", Test_RMSE_linear)



regressor = Lasso()  
regressor.fit(modified_X_Train, y_Train)
Train_RMSE_lasso = get_error(y_Train, regressor.predict(modified_X_Train))
Test_RMSE_lasso = get_error(y_Test, regressor.predict(modified_X_Test))

print("Train RMSE in lasso: ", Train_RMSE_lasso)
print("Test RMSE in lasso: ", Test_RMSE_lasso)


# In[2]:


def get_error(y, y_Pred):
    mse = metrics.mean_squared_error(y, y_Pred)
    rmse = math.sqrt(mse)
    return rmse


# In[ ]:




