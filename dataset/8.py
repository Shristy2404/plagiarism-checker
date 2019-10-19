#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')

x_train=dataset_train['Father'].values.reshape(-1,1)
y_train=dataset_train['Son'].values.reshape(-1,1)

x_test=dataset_test['Father'].values.reshape(-1,1)
y_test=dataset_test['Son'].values.reshape(-1,1)

poly=PolynomialFeatures(degree=10)
modified_x=poly.fit_transform(x_train)
modifiedtest_x=poly.fit_transform(x_test)


# In[11]:


#Lasso
train_err=[]
test_err=[]
diff_err=[]
poly_11=PolynomialFeatures(degree=10)
modified_x=poly.fit_transform(x_train)
modifiedtest_x=poly.fit_transform(x_test)
alpha_vals = np.linspace(0,1,9)
for alpha_v in alpha_vals:
    reg=Lasso(alpha=alpha_v)
    reg.fit(modified_x,y_train)
    
    trainRMSE=math.sqrt(mean_squared_error(y_train,reg.predict(modified_x)))
    testRMSE=math.sqrt(mean_squared_error(y_test,reg.predict(modifiedtest_x)))
    train_err.append(trainRMSE)
    test_err.append(testRMSE)
    diff_err.append(trainRMSE-testRMSE)
print("At alpha :\n",alpha_vals)
print("RMSE Training set :\n",train_err)
print("RMSE Test Set :\n",test_err)
print("Difference of Train and Test Errors:\n",diff_err)
plt.title("Ridge")
plt.xlabel('Alpha Value')
plt.ylabel('RMSE')
plt.plot(np.linspace(0,1,9),train_err,'bo-',label="Train")
plt.plot(np.linspace(0,1,9),test_err,'ro-',label="Test")
plt.legend()
plt.show()


# In[ ]:




