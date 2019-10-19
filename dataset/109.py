#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd 
import numpy as np 
from math import sqrt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


# In[8]:


traindata = pd.read_csv("PartD_training.csv")
testdata = pd.read_csv("PartD_testing.csv")


# In[9]:


x_train = traindata['Father'].values.reshape(-1,1)
y_train = traindata['Son'].values.reshape(-1,1)
x_test=testdata['Father'].values.reshape(-1,1)
y_test=testdata['Son'].values.reshape(-1,1)


# In[10]:


polyreg = PolynomialFeatures(degree=10)
modified_x_train = polyreg.fit_transform(x_train)
modified_x_test = polyreg.fit_transform(x_test)


# In[11]:


model = Lasso()
model.fit(modified_x_train, y_train)

y_predicted_test = model.predict(modified_x_test)
y_predicted_train = model.predict(modified_x_train)


# In[12]:


print('RMSE Train:',sqrt(mean_squared_error(y_train, y_predicted_train)))
print('RMSE Test:',sqrt(mean_squared_error(y_test, y_predicted_test)))


# In[ ]:




