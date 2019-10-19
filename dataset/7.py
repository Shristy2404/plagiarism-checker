#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


traindata=pd.read_csv('C:\\Users\\saranga.bordoloi\\BITS\\train.csv')
testdata=pd.read_csv('C:\\Users\\saranga.bordoloi\\BITS\\test.csv')

x_train=traindata['Father'].values.reshape(-1,1)
y_train=traindata['Son'].values.reshape(-1,1)
x_test=testdata['Father'].values.reshape(-1,1)
y_test=testdata['Son'].values.reshape(-1,1)

poly=PolynomialFeatures(degree=10)
modified_x_train= poly.fit_transform(x_train)
modified_x_test= poly.fit_transform(x_test)


# In[3]:


reg = Lasso(alpha=0.5)
reg.fit(modified_x_train, y_train)


# In[4]:


print('Lasso Train RMSE: ', math.sqrt(mean_squared_error(y_train, reg.predict(modified_x_train))))
print('Lasso Test RMSE: ', math.sqrt(mean_squared_error(y_test, reg.predict(modified_x_test))))


# In[6]:


reg = Ridge(alpha=0.5)
reg.fit(modified_x_train, y_train)
print('Ridge Train RMSE: ', math.sqrt(mean_squared_error(y_train, reg.predict(modified_x_train))))
print('Ridge Test RMSE: ', math.sqrt(mean_squared_error(y_test, reg.predict(modified_x_test))))


# In[8]:


#Lasso

train_err = []
test_err = []

poly = PolynomialFeatures(degree=10)
modified_x_train = poly.fit_transform(x_train)
modified_x_test= poly.fit_transform(x_test)

alpha_vals = np.linspace(0,1,10)
for alpha_v in alpha_vals:
    reg=Lasso(alpha=alpha_v)
    reg.fit(modified_x_train, y_train)
    
    train_err.append(math.sqrt(mean_squared_error(y_train, reg.predict(modified_x_train))))
    test_err.append(math.sqrt(mean_squared_error(y_test, reg.predict(modified_x_test))))

plt.title('Lasso')
plt.xlabel('Alpha Value')
plt.ylabel('RMSE')
plt.plot(np.linspace(0,1,10), train_err, 'bo-', label='Train')
plt.plot(np.linspace(0,1,10), test_err, 'ro-', label='Test')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




