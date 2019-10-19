#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression with OLS
# 
# The notebook contains incomplete code for the problem along with the necessary information in the form of comments that helps you to complete the project.  
# 
# ## Step 1 - Importing the required libraries 
# 
# We have completed this step for you. Please go through it to have a clear idea about the libraries that are used in the project.

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# Remember what this line did?
get_ipython().run_line_magic('matplotlib', 'inline')
import math 

# the library we will use to create the model 
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[3]:


# 2.1 Read the dataset from the csv file using pandas and write into a pandas dataframe object named 'dataset'
train_dataset=pd.read_csv('project - part D - training data set.csv')
test_dataset=pd.read_csv('project - part D - testing data set.csv')


# In[4]:


x_train = train_dataset['Father'].values.reshape(-1,1)
y_train = train_dataset['Son'].values.reshape(-1,1)
x_test = test_dataset['Father'].values.reshape(-1,1)
y_test = test_dataset['Son'].values.reshape(-1,1)


# In[7]:


poly = PolynomialFeatures(10)
x_train_ = poly.fit_transform(x_train)
x_test_ = poly.fit_transform(x_test)
ls=Lasso()
ls.fit(x_train_,y_train)
y_pred=ls.predict(x_test_)
print("TRAIN RMSE :", math.sqrt(mean_squared_error(y_train,ls.predict(x_train_))))
print("TEST RMSE :", math.sqrt(mean_squared_error(y_test,y_pred)))


# For a typical regression model, the mean squared error for degree 10 for the test data was around 1.8. But for the lasso regression model, the mean squared error for degree 10 for the test data came down to 1.54
