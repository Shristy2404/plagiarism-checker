#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
import warnings
warnings.filterwarnings('ignore')

traindata=pd.read_csv("project - part D - training data set.csv")
testdata=pd.read_csv("project - part D - testing data set.csv")
x_train=traindata['Father'].values.reshape(-1,1)
y_train=traindata['Son'].values.reshape(-1,1)
x_test=testdata['Father'].values.reshape(-1,1)
y_test=testdata['Son'].values.reshape(-1,1)


poly=PolynomialFeatures(degree=10)
modified_X_train=poly.fit_transform(x_train)
modified_X_test=poly.fit_transform(x_test)
reg=Lasso()
reg.fit(modified_X_train,y_train)
print('Lasso Train RMSE for degree 10 : ', math.sqrt(mean_squared_error(y_train,reg.predict(modified_X_train))))
print('Lasso Test RMSE for degree 10 : ', math.sqrt(mean_squared_error(y_test,reg.predict(modified_X_test))))


# In[ ]:




