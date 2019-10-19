#!/usr/bin/env python
# coding: utf-8

# In[144]:


import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import metrics
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings("ignore")


# In[145]:


data_train = pd.read_csv("project - part D - training data set.csv")
data_test = pd.read_csv("project - part D - testing data set.csv")


# In[146]:


x_train = data_train['Father'].values.reshape(-1,1)
y_train = data_train['Son'].values.reshape(-1,1)
x_test = data_test['Father'].values.reshape(-1,1)
y_test = data_test['Son'].values.reshape(-1,1)


# In[147]:


poly = PolynomialFeatures(degree = 10) 
modified_X_train = poly.fit_transform(x_train)
modified_X_test = poly.fit_transform(x_test)
reg = Lasso()
reg1 = reg.fit(modified_X_train,y_train)
y_train_predicted = reg1.predict(modified_X_train)
y_test_predicted =  reg1.predict(modified_X_test)
train_error=math.sqrt(metrics.mean_squared_error(y_train, y_train_predicted))
test_error=math.sqrt(metrics.mean_squared_error(y_test, y_test_predicted))
print("train_rmse for degree 10:", train_error)
print("test_rmse for degree 10:" , test_error)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




