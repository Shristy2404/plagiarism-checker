#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures 

from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error
import math 


# In[16]:


df = pd.read_csv('partD-training_dataset.csv')
X = df['Father'].values
X = X.reshape(-1,1)
y = df['Son'].values.reshape(-1,1)


# In[17]:


df = pd.read_csv('partD-testing_dataset.csv')
X_test = df['Father'].values
X_test = X_test.reshape(-1,1)
y_test = df['Son'].values.reshape(-1,1)


# In[19]:


poly = PolynomialFeatures(degree=10)
modified_X = poly.fit_transform(X)


# In[22]:


# Lasso Regression With Default Alpha/ Lamda value 
reg = Lasso()
reg.fit(X,y)

print('Lasso Trian RMSE: ',math.sqrt(mean_squared_error(y,reg.predict(X))))
print('Lasso Test RMSE: ',math.sqrt(mean_squared_error(y_test,reg.predict(X_test))))


# In[21]:


# Lasso Regression for varying degree from 1 to 10
train_err =[]
test_err = []

for i in range(1,10):
    poly = PolynomialFeatures(degree=i)
    modified_X = poly.fit_transform(X)
    
alpha_vals = np.linspace(1,5,10)
for alpha_v in alpha_vals:
    reg = Lasso(alpha=alpha_v)
    reg.fit(X,y)
    
    train_err.append(math.sqrt(mean_squared_error(y,reg.predict(X))))
    test_err.append(math.sqrt(mean_squared_error(y_test,reg.predict(X_test))))
    
    
print('Train RMSE',train_err)
print('Test RMSE',test_err)
plt.title('Lasso')
plt.xlabel('Alpha Value')
plt.ylabel('RMSE')
plt.plot(np.linspace(1,10,10),train_err,'bo-', label='Train')
plt.plot(np.linspace(1,10,10),test_err,'ro-', label='Test')
plt.legend()
plt.savefig ('2018wilp608_Lasso_part_d_plot.png')
plt.show


# In[ ]:





# In[ ]:




