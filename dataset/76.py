#!/usr/bin/env python
# coding: utf-8

# # Multi Variable Linear Regression: Lasso regression for handling ovefitting 

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import math


# # Step 2 - Reading the dataset and processing 

# In[2]:


df1=pd.read_csv('project - part D - training data set.csv')
df1.drop(df1.columns[df1.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
df1.rename(columns={"'	Father	'": "Father"})


# In[3]:


df2=pd.read_csv('project - part D - testing data set.csv')
df2.drop(df2.columns[df2.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
df2.rename(columns={"'	Father	'": "Father"})


# In[4]:


X_train=df1['Father'].values.reshape(-1,1)
y_train=df1['Son'].values.reshape(-1,1)
X_test=df2['Father'].values.reshape(-1,1)
y_test=df2['Son'].values.reshape(-1,1)


# In[10]:


poly=PolynomialFeatures(degree=10)
modified_X_train=poly.fit_transform(X_train)
modified_X_test=poly.fit_transform(X_test)
reg=Lasso()
reg.fit(modified_X_train,y_train)
print('Train RMSE: ',math.sqrt(mean_squared_error(y_train,reg.predict(modified_X_train))))
print('Test RMSE: ',math.sqrt(mean_squared_error(y_test,reg.predict(modified_X_test))))
print("Regression Coefficients are:",reg.coef_)
print("Regression Intercept is:",reg.intercept_)


# # Step 3 Fitting the model with varying regularization parameters  and generating error

# In[7]:


train_err=[]
test_err=[]
poly_10=PolynomialFeatures(degree = 10)
modified_X_train=poly_10.fit_transform(X_train)
modified_X_test=poly_10.fit_transform(X_test)
alpha_vals=np.linspace(0,1,10)
for alpha_v in alpha_vals:
    reg=Lasso(alpha=alpha_v)
    reg.fit(modified_X_train,y_train)
    print(reg.coef_)
    print(reg.intercept_)
    train_err.append(math.sqrt(mean_squared_error(y_train,reg.predict(modified_X_train))))
    test_err.append(math.sqrt(mean_squared_error(y_test,reg.predict(modified_X_test))))
print(train_err)
print(test_err)
plt.xlabel('Lasso factor')
plt.ylabel('RMSE')
plt.plot(np.linspace(0,1,10),train_err,'bo-',label='Train',color='g')
plt.plot(np.linspace(0,1,10),test_err,'bo-',label='Test',color='r')
plt.legend()
plt.show


# In[ ]:




