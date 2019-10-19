#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression with Stochastic Gradient Descent

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn import metrics 
import warnings as wn
wn.filterwarnings('ignore')


# # Lasso

# In[3]:


train = pd.read_csv("project - part D - training data set.csv")
test = pd.read_csv("project - part D - testing data set.csv")
x_train = train['Father'].values.reshape(-1,1)
y_train = train['Son'].values.reshape(-1,1)
x_test = test['Father'].values.reshape(-1,1)
y_test = test['Son'].values.reshape(-1,1)


# In[4]:


rmse_poly_train = []
def predict_train():
    predict_poly_train = model.predict(modified_X_train)
    mse_poly_train = metrics.mean_squared_error(y_train, predict_poly_train)
    rmse_poly_train.append(np.sqrt(mse_poly_train))


# In[5]:


rmse_poly_test = []
def predict_test():
    predict_poly_test = model.predict(modified_X_test)
    mse_poly_test = metrics.mean_squared_error(y_test, predict_poly_test)
    rmse_poly_test.append(np.sqrt(mse_poly_test))


# In[6]:


for i in range(10,11):
    poly_features = PolynomialFeatures(i)
    modified_X_train = poly_features.fit_transform(x_train)
    modified_X_test = poly_features.fit_transform(x_test)
    model = Lasso() 
    model.fit(modified_X_train, y_train)
    predict_train()
    predict_test()


# In[7]:


print("rmse for train: " + str(rmse_poly_train))
print("rmse for test: " + str(rmse_poly_test))
print("\n")
# In[8]:


print("Comparing this rmse " + str(min(rmse_poly_test)) + " with linear regression model it is evident that Lasso model with 10 polynomial has similar error as that of 1st degree polynomial in linear regression model. So, Lasso model gives us very good polynomial")


# In[ ]:




