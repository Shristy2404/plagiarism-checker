#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math


# In[23]:


TestDataset = pd.read_csv('project - part D - testing data set.csv')
TrainDataset = pd.read_csv('project - part D - training data set.csv')


# In[24]:


Xtrain = TrainDataset['Father']
Xtrain = Xtrain.values.reshape(-1,1)

Ytrain = TrainDataset['Son']
Ytrain = Ytrain.values.reshape(-1,1)

Xtest = TestDataset['Father']
Xtest = Xtest.values.reshape(-1,1)

Ytest = TestDataset['Son']
Ytest = Ytest.values.reshape(-1,1)


# In[25]:


poly = PolynomialFeatures(degree=10)
ModifiedXtrain = poly.fit_transform(Xtrain)
ModifiedXtest = poly.fit_transform(Xtest)


# In[26]:


RegLin = linear_model.LinearRegression()
RegLin.fit(ModifiedXtrain,Ytrain)
sons_height_predicted_Lin_train = RegLin.predict(ModifiedXtrain)
sons_height_predicted_Lin_test = RegLin.predict(ModifiedXtest)

RegLasso = Lasso()
RegLasso.fit(ModifiedXtrain,Ytrain)
sons_height_predicted_Lasso_train = RegLasso.predict(ModifiedXtrain)
sons_height_predicted_Lasso_test = RegLasso.predict(ModifiedXtest)


# In[27]:


RMSE_Lin_train = math.sqrt(mean_squared_error(Ytrain,sons_height_predicted_Lin_train))
RMSE_Lin_test = math.sqrt(mean_squared_error(Ytest,sons_height_predicted_Lin_test))

RMSE_Lasso_train = math.sqrt(mean_squared_error(Ytrain,sons_height_predicted_Lasso_train))
RMSE_Lasso_test = math.sqrt(mean_squared_error(Ytest,sons_height_predicted_Lasso_test))


# In[28]:


print("RMSE for Normal Linear Regression for training data", RMSE_Lin_train)
print("RMSE for Normal Linear Regression for test data", RMSE_Lin_test)

print("\n\nRMSE for Lasso for training data" , RMSE_Lasso_train)
print("RMSE for Lasso for test data", RMSE_Lasso_test)
print("Improvement in RMSE for test data in Lasso as compared to normal linear regression is:", RMSE_Lin_test-RMSE_Lasso_test )


# In[ ]:




