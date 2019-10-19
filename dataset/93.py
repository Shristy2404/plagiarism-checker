#!/usr/bin/env python
# coding: utf-8

# ## Lasso Regression
# 
# The notebook contains code to build the Lasso Regression
# 
# ## Step 1 - Importing the required libraries 

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# Remember what this line did?
get_ipython().run_line_magic('matplotlib', 'inline')
import math
from math import sqrt

# the library we will use to create the model 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# ## Step 2 - Reading the dataset and splitting it into testing and training data

# In[3]:


# Read the training ataset from the csv file using pandas and write into a pandas dataframe object named 'train_data'
train_data = pd.read_csv('project - part D - training data set.csv')

# Read the test ataset from the csv file using pandas and write into a pandas dataframe object named 'test_data'
test_data = pd.read_csv('project - part D - testing data set.csv')


# In[4]:


# Get  the dependent and independent variables
x_train = train_data['Father'].values.reshape(-1,1)
y_train = train_data['Son'].values.reshape(-1,1)

x_test = test_data['Father'].values.reshape(-1,1)
y_test = test_data['Son'].values.reshape(-1,1)


# ## Step 3 - Generating the Regression model using Lasso and Typical Regression

# In[7]:


# generate a model object using Lasso Regression

poly = PolynomialFeatures(degree=10)
modified_xTrain = poly.fit_transform(x_train)
modified_xTest = poly.fit_transform(x_test)

lasso_reg = Lasso(alpha=1.0,max_iter=10000)
lasso_reg.fit(modified_xTrain,y_train)

#print('Coefficients using Lasso: ',lasso_reg.coef_)
#print('Intercept using Lasso: ',lasso_reg.intercept_)
print('RMSE for Train Data using Lasso: ',sqrt(mean_squared_error(y_train,lasso_reg.predict(modified_xTrain))))
print('RMSE for Test Data using Lasso: ',sqrt(mean_squared_error(y_test,lasso_reg.predict(modified_xTest))))


# In[8]:


# generate a model object using Typical Regression

reg = LinearRegression()
reg.fit(modified_xTrain,y_train)

#print('Coefficients using Typical Regression Model: ',reg.coef_)
#print('Intercept using Typical Regression Model: ',reg.intercept_)
print('RMSE for Train Data using Lasso: ',sqrt(mean_squared_error(y_train,reg.predict(modified_xTrain))))
print('RMSE for Test Data using Lasso: ',sqrt(mean_squared_error(y_test,reg.predict(modified_xTest))))


# In[ ]:




