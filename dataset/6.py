#!/usr/bin/env python
# coding: utf-8

# # Lasso Regression

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math


# In[31]:


traindata = pd.read_csv('project - part D - training data set.csv')
testdata = pd.read_csv('project - part D - testing data set.csv')


# In[32]:


x_train = traindata['Father'].values.reshape(-1,1)
y_train = traindata['Son'].values.reshape(-1,1)


# In[33]:


x_test = testdata['Father'].values.reshape(-1,1)
y_test = testdata['Son'].values.reshape(-1,1)


# In[41]:


polyreg = PolynomialFeatures(degree=10) 
x_modified_train = polyreg.fit_transform(x_train)
x_modified_test = polyreg.fit_transform(x_test)


# In[42]:


reg = Lasso(alpha = 0.5).fit(x_modified_train,y_train)

print('Lasso Train RMSE:', math.sqrt(mean_squared_error(y_train, reg.predict(x_modified_train))))
print('Lasso Test RMSE:', math.sqrt(mean_squared_error(y_test, reg.predict(x_modified_test))))


# In[43]:


trainerror = []
testerror = []
alpha_vals = np.linspace(0,1,10)

for alpha_v in alpha_vals:
    
    reg = Lasso(alpha = alpha_v).fit(x_train,y_train)
    
    y_predicted_test = reg.predict(x_test)
    y_predicted_train = reg.predict(x_train)
    
    trainerror.append(math.sqrt(mean_squared_error(y_train, y_predicted_train)))
    testerror.append(math.sqrt(mean_squared_error(y_test, y_predicted_test)))
    
print('Training_points:',x_train.shape[0])
print('Testing_points:',x_test.shape[0])


# In[44]:


plt.title('Lasso')
plt.xlabel('Alpha value')
plt.ylabel('RMSE')

plt.plot(np.linspace(0,1,10),trainerror,'bX-', label= 'Train')
plt.plot(np.linspace(0,1,10),testerror,'rX-', label= 'Test')

plt.legend()
plt.show()


# # DONE!
