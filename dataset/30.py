#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
import os.path

from sklearn.metrics import mean_squared_error
import math


# In[16]:


trainFileName = 'project - part D - training data set.csv'
testFileName = 'project - part D - testing data set.csv'
trainFileExist = os.path.isfile(trainFileName)
testFileExist = os.path.isfile(testFileName)

if not trainFileExist:
    trainFileName = 'train.csv'
    
if not testFileExist:
    testFileName = 'test.csv'
  

dfTrain = pd.read_csv(trainFileName)
yTrain = dfTrain['Son'].values.reshape(-1, 1)
xTrain = dfTrain['Father'].values.reshape(-1, 1)

dfTest = pd.read_csv(testFileName)
yTest = dfTest['Son'].values.reshape(-1, 1)
xTest = dfTest['Father'].values.reshape(-1, 1)


# In[17]:


lassoReg = Lasso()

poly = PolynomialFeatures(degree = 10)
modified_x_train = poly.fit_transform(xTrain)
modified_x_test = poly.fit_transform(xTest)
lassoReg.fit(modified_x_train, yTrain)
trainRMSE = math.sqrt(mean_squared_error(yTrain, lassoReg.predict(modified_x_train)))
testRMSE = math.sqrt(mean_squared_error(yTest, lassoReg.predict(modified_x_test)))

    
print('train RMSE: ', trainRMSE)
print ('test RMSE: ', testRMSE)


# In[ ]:




