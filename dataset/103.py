#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# the library we will use to create the model 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import PolynomialFeatures 
import math
import sklearn.metrics as metrics


# In[5]:


dataset=pd.read_csv("project - part D - training data set.csv")
X_train= dataset['Father'].values.reshape(-1,1)
y_train = dataset['Son'].values.reshape(-1,1)
dataset1=pd.read_csv("project - part D - testing data set.csv")
X_test= dataset1['Father'].values.reshape(-1,1)
y_test = dataset1['Son'].values.reshape(-1,1)


# In[6]:


poly=PolynomialFeatures(degree=10)
modified_X_train=poly.fit_transform(X_train)
modified_X_test=poly.fit_transform(X_test)


# In[20]:


#RMSE for Lasso
reg=Lasso(alpha=0.5)
reg.fit(modified_X_train, y_train)
y_predicted_test=reg.predict(modified_X_test)
y_predicted_train=reg.predict(modified_X_train)
print('Lasso RMSE Train:',math.sqrt(mean_squared_error(y_train, y_predicted_train)))
print('Lasso RMSE Test:',math.sqrt(mean_squared_error(y_test, y_predicted_test)))


# In[21]:


#RMSE for regular/Typical Regression
regTypical=LinearRegression()
regTypical.fit(modified_X_train,y_train)
y_predicted_test=regTypical.predict(modified_X_test)
y_predicted_train=regTypical.predict(modified_X_train)
print('Typical RMSE Train:',math.sqrt(mean_squared_error(y_train, y_predicted_train)))
print('Typical RMSE Test:',math.sqrt(mean_squared_error(y_test, y_predicted_test)))


# As when comapred to Typical regression the lasso regression RSME for training is slightly more but the lasso regression RSME for testing is less than the Typical regression RMSE

# In[ ]:




