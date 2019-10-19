#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math


# In[4]:


d1=pd.read_csv('project - part D - training data set.csv')
x_train = d1['Father'].values.reshape(-1,1)
y_train = d1['Son'].values.reshape(-1,1)
d2=pd.read_csv('project - part D - testing data set.csv')
x_test = d2['Father'].values.reshape(-1,1)
y_test = d2['Son'].values.reshape(-1,1)

poly=PolynomialFeatures(degree=10)
Modified_x_train=poly.fit_transform(x_train)
Modified_x_test=poly.fit_transform(x_test)
reg=Lasso(alpha=1)
reg.fit(Modified_x_train,y_train)

print('Lasso Train error:' ,math.sqrt(mean_squared_error(y_train,reg.predict(Modified_x_train))))
print('Lasso Test error:' ,math.sqrt(mean_squared_error(y_test,reg.predict(Modified_x_test))))


# In[ ]:





# In[ ]:




