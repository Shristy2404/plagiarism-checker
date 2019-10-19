#!/usr/bin/env python
# coding: utf-8

# # LASSO REGRESSION

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn import metrics 


# In[6]:


datasetTrain = pd.read_csv('project - part D - training data set.csv')
xTrain = datasetTrain['Father'].values.reshape(-1,1)
yTrain = datasetTrain ['Son'].values.reshape(-1,1)

datasetTest = pd.read_csv('project - part D - testing data set.csv')
xTest = datasetTest['Father'].values.reshape(-1,1)
yTest = datasetTest ['Son'].values.reshape(-1,1)


# In[7]:


poly = PolynomialFeatures(degree=10)
modified_xTrain = poly.fit_transform(xTrain)
modified_xTest = poly.fit_transform(xTest)

alphas =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
x_axis= alphas
y_axis =[[],[]]

for a in alphas:
    
    reg = Lasso(alpha=a,max_iter=1000)
    reg.fit(modified_xTrain,yTrain)
    rMSE_Train = sqrt(mean_squared_error(yTrain,reg.predict(modified_xTrain)))
    rMSE_Test = sqrt(mean_squared_error(yTest,reg.predict(modified_xTest)))

    print('Coeffitient for alpha:',a,'=',reg.coef_)
    print('Intercept for alpha:',a,'=',reg.intercept_)
    print('Train RMSE for alpha:',a,'=',rMSE_Train)
    print('Test RMSE for alpha:',a,'=',rMSE_Test)
    print('_________________________________________________________________________')

    y_axis[0].append(rMSE_Train)
    y_axis[1].append(rMSE_Test)


# In[8]:


plt.xlabel('Alphas')
plt.ylabel('RMSE')
plt.plot(x_axis,y_axis[0],'bo-',label='Training Data Set')
plt.plot(x_axis,y_axis[1],'ro-',label='Testing Data Set')
plt.legend()
plt.show() 


# ### CONCLUSION - FROM PLOT WE CAN CONCLUDE THAT FOR POLYNOMIAL OF DEGREE 10, LASSO Regression Model for Alpha=1.0  GIVES BEST MODEL WITH RMSE = 1.54177263024972
