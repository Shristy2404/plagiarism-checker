#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model, model_selection,metrics,preprocessing
import math
import warnings
warnings.filterwarnings("ignore")


# In[2]:


train = pd.read_csv("project - part D - training data set.csv")
test = pd.read_csv("project - part D - testing data set.csv")


# In[3]:


lasso = linear_model.Lasso()


# In[4]:


train_modified = pd.DataFrame(
    preprocessing.PolynomialFeatures(10).fit_transform(train[['Father']])
)
test_modified  = pd.DataFrame(
    preprocessing.PolynomialFeatures(10).fit_transform(test[['Father']])
)


# In[5]:


lasso.fit(train_modified,train[['Son']])


# In[6]:


mse_train = metrics.mean_squared_error(train[['Son']],lasso.predict(train_modified))
rmse_train = math.sqrt(mse_train)
print("RMSE for Train Data: " ,rmse_train)


# In[7]:


mse_test = metrics.mean_squared_error(test[['Son']],lasso.predict(test_modified))
rmse_test = math.sqrt(mse_test)
print("RMSE for Test Data: ",rmse_test)

