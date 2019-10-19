#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn import metrics


# In[34]:


train_dataset = pd.read_csv('project - part D - training data set.csv')
test_dataset = pd.read_csv('project - part D - testing data set.csv')

train_dataset = train_dataset.drop('Unnamed: 0',axis=1)
test_dataset = test_dataset.drop('Unnamed: 0',axis=1)

x_train = train_dataset['Father'].values.reshape(-1,1)
y_train = train_dataset['Son'].values.reshape(-1,1)
x_test = test_dataset['Father'].values.reshape(-1,1)
y_test = test_dataset['Son'].values.reshape(-1,1)


# In[35]:


# Using include_bias as false to ignore bias column, in which all polynomial powers are zero
poly = PolynomialFeatures(degree=10,include_bias=False)
modified_x_train = poly.fit_transform(x_train)
modified_x_test = poly.fit_transform(x_test)

#Lasso regression using default alpha.
reg = Lasso()
reg.fit(modified_x_train, y_train)
train_error = math.sqrt(mean_squared_error(y_train,reg.predict(modified_x_train)))
test_error = math.sqrt(mean_squared_error(y_test,reg.predict(modified_x_test)))
print('Lasso Regression Training RMSE: ', train_error)
print('Lasso Regression Test RMSE: ', test_error)

