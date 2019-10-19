#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression,Lasso,Ridge
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

df1=pd.read_csv('project - part D - training data set.csv')
df2=pd.read_csv('project - part D - testing data set.csv')
X1=df1.iloc[:,1].values.reshape(-1,1)
y1=df1.iloc[:,2].values.reshape(-1,1)
X2=df2.iloc[:,1].values.reshape(-1,1)
y2=df2.iloc[:,2].values.reshape(-1,1)

poly=PolynomialFeatures(degree=10)

X_train=poly.fit_transform(X1)
X_test=poly.fit_transform(X2)
y_train=y1
y_test=y2



#Lasso
lasso=Lasso(alpha=1)
lasso.fit(X_train,y_train)
y_predict_train=lasso.predict(X_train)
y_predict_test=lasso.predict(X_test)


MSE1=metrics.mean_squared_error(y_train,y_predict_train)
RMSE1=math.sqrt(MSE1)

    
MSE2=metrics.mean_squared_error(y_test,y_predict_test)
RMSE2=math.sqrt(MSE2)


print('training error is :',RMSE1)
print('testing error is :',RMSE2)


# In[ ]:




