#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


# In[16]:


training_dataset = pd.read_csv('project - part D - training data set.csv')
xTrain = training_dataset['Father'].values.reshape(-1,1)
yTrain = training_dataset['Son'].values.reshape(-1,1)

testing_dataset = pd.read_csv('project - part D - testing data set.csv')
xTest = testing_dataset['Father'].values.reshape(-1,1)
yTest = testing_dataset['Son'].values.reshape(-1,1)


# In[45]:


poly = PolynomialFeatures(degree=10)
modified_xTrain = poly.fit_transform(xTrain)
modified_xTest = poly.fit_transform(xTest)

train_err = []
test_err = []
lamda_vals = []
reg_weights = []
reg = Lasso(normalize=True)

for i in range(1, 11, 1):
    lamda = i * 0.01
    lamda_vals.append(lamda)
    reg.alpha = lamda #, tol=0.00000001, max_iter=10000)
    reg.fit(modified_xTrain, yTrain)
    reg_weights.append(reg.coef_)
    train_error = math.sqrt(mean_squared_error(yTrain, reg.predict(modified_xTrain)))
    test_error = math.sqrt(mean_squared_error(yTest, reg.predict(modified_xTest)))
    #print('For lamda=',lamda,': train_error=', train_error)
    #print('For lamda=',lamda,': test_error=', test_error)
    train_err.append(math.sqrt(mean_squared_error(yTrain, reg.predict(modified_xTrain))))
    test_err.append(math.sqrt(mean_squared_error(yTest, reg.predict(modified_xTest)))) 


# In[46]:


plt.title('Lasso Regression: RMSE vs Lamda')
plt.xlabel('Lamda')
plt.ylabel('RMSE')
plt.plot(np.linspace(0.01,0.1,10), train_err, 'bo-', label='Train')
plt.plot(np.linspace(0.01,0.1,10), test_err, 'ro-', label='Test')
plt.legend()
plt.show()   


# In[47]:


# Without lasso and Ridge: Polynomial regression for degree 10
#Train RSME error for polynomial degree 10:  1.3787128183638069
#Test RSME error for polynomial degree 10::  1.8179720763695844


# In[48]:


# Best lasso model with lamda for least RMSE on testing dataset
min_index = test_err.index(min(test_err))
print('Lasso Regression with polynomial degree = 10 and lamda = ', lamda_vals[min_index])
print(' Weights:', reg_weights[min_index].flatten())
print(' Training RMSE:', train_err[min_index])
print(' Testing RMSE:', test_err[min_index])
print('NOTE: In lasso, most of the model params are zero.')


# In[ ]:




