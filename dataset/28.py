#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso


# In[38]:


training_dataset= pd.read_csv('project - part D - training data set.csv')
x= training_dataset['Father']
#x/= 1000 # feature engineering
x_train= x.values.reshape(-1,1)
y_train= training_dataset['Son'].values.reshape(-1,1)


# In[39]:


testing_dataset= pd.read_csv('project - part D - testing data set.csv')
f= testing_dataset['Father']
#f/= 1000 # feature engineering
x_test= f.values.reshape(-1,1)
y_test= testing_dataset['Son'].values.reshape(-1,1)


# In[40]:


poly= PolynomialFeatures(degree=10)
modified_x_train=poly.fit_transform(x_train)
modified_x_test=poly.fit_transform(x_test)
reg = Lasso(alpha=1.0,max_iter=10000)
reg.fit(modified_x_train,y_train)
train_error= math.sqrt(metrics.mean_squared_error(y_train,reg.predict(modified_x_train)))
test_error= math.sqrt(metrics.mean_squared_error(y_test,reg.predict(modified_x_test)))


# In[41]:


print('RMSEs for Lasso regression of degree 10:')
print('RMSE for training data:',train_error)
print('RMSE for test data:',test_error)


# In[42]:


linear_reg = LinearRegression()
linear_reg.fit(modified_x_train,y_train)
linear_train_error= math.sqrt(metrics.mean_squared_error(y_train,linear_reg.predict(modified_x_train)))
linear_test_error= math.sqrt(metrics.mean_squared_error(y_test,linear_reg.predict(modified_x_test)))
print('RMSEs for polynomial regression with degree 10:')
print('RMSE for training data:',linear_train_error)
print('RMSE for test data:',linear_test_error)


