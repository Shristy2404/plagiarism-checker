# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 22:11:43 2019

@author: arunc
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# Remember what this line did?
#get_ipython().run_line_magic('matplotlib', 'inline')
import math 

# the library we will use to create the model 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

masterTrainData = pd.read_csv('project - part D - training data set.csv')
masterTestData = pd.read_csv('project - part D - testing data set.csv')

XFTrain = masterTrainData['Father'].values/100
XFTrain = XFTrain.reshape(-1,1)
YSTrain = masterTrainData['Son'].values.reshape(-1,1)

XFTest = masterTestData['Father'].values/100
XFTest = XFTest.reshape(-1,1)
YSTest = masterTestData['Son'].values.reshape(-1,1)

poly = PolynomialFeatures(degree=10)
XFTrain_m = poly.fit_transform(XFTrain)
XFTest_m = poly.fit_transform(XFTest)

lRgr = LinearRegression()
lRgr.fit(XFTrain_m, YSTrain)

errorLnrRMSETrain=((mean_squared_error(YSTrain,lRgr.predict(XFTrain_m)))**0.5)
errorLnrRMSETest=((mean_squared_error(YSTest,lRgr.predict(XFTest_m)))**0.5)

lassoMdl = Lasso()
lassoMdl.fit(XFTrain_m, YSTrain)

errorLassoRMSETrain=((mean_squared_error(YSTrain,lassoMdl.predict(XFTrain_m)))**0.5)
errorLassoRMSETest=((mean_squared_error(YSTest,lassoMdl.predict(XFTest_m)))**0.5)

if errorLassoRMSETest<errorLnrRMSETest:
    print('Error on Lasso is Smaller. Lasso RMSE:', errorLassoRMSETest,' Linear RMSE:',errorLnrRMSETest)
else:
    print('Error on Linear Model is Smaller. Lasso RMSE:', errorLassoRMSETest,' Linear RMSE:',errorLnrRMSETest)
        
    
    

#for i in range(10):
    





