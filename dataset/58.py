#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


# In[3]:


df_train  = pd.read_csv('Training_data_set.csv')
df_test  = pd.read_csv('Testing_data_set.csv')
x_train = df_train['Father'].values.reshape(-1,1)
y_train = df_train['Son'].values.reshape(-1,1)
x_test = df_test['Father'].values.reshape(-1,1)
y_test = df_test['Son'].values.reshape(-1,1)


# In[4]:


poly = PolynomialFeatures(degree=11)
modified_x_train = poly.fit_transform(x_train)
modified_x_test = poly.fit_transform(x_test)


# In[17]:


reg = Lasso(alpha=0.5)
reg.fit(x_train,y_train)


# In[18]:


print("Lasso Train RMSE : ", math.sqrt(mean_squared_error(y_train,reg.predict(x_train))))
print("Lasso Test RMSE : ", math.sqrt(mean_squared_error(y_test,reg.predict(x_test))))


# In[ ]:




