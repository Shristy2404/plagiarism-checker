#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error
import math


# In[12]:


poly = PolynomialFeatures(degree=10)


# In[13]:


df_train = pd.read_csv("project - part D - training data set.csv")
X_train = poly.fit_transform(df_train['Father'].values.reshape(-1, 1))
y_train = df_train['Son'].values.reshape(-1, 1)


# In[14]:


df_test = pd.read_csv("project - part D - testing data set.csv")
X_test = poly.fit_transform(df_test['Father'].values.reshape(-1, 1))
y_test = df_test['Son'].values.reshape(-1, 1)


# In[15]:


reg = Lasso()
reg.fit(X_train, y_train)

print('Coefficient is ', reg.coef_)
print('Intercept is ', reg.intercept_)
print('Lasso Train RMSE: ', math.sqrt(mean_squared_error(y_train, reg.predict(X_train))))
print('Lasso Test RMSE: ', math.sqrt(mean_squared_error(y_test, reg.predict(X_test))))


# In[ ]:




