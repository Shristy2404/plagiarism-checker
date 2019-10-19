#!/usr/bin/env python
# coding: utf-8

# ## Lasso Regression
# 

# In[36]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 


# In[37]:


train_df = pd.read_csv('project - part D - training data set.csv')
X_train = train_df['Father'].values
X_train = X_train.reshape(-1,1)
y_train = train_df['Son'].values.reshape(-1,1)


# In[38]:


test_df = pd.read_csv('project - part D - testing data set.csv')
X_test = test_df['Father'].values
X_test = X_test.reshape(-1,1)
y_test = test_df['Son'].values.reshape(-1,1)


# ### Find Lasso and Polynomial regression RMSE for degree 10

# In[39]:


poly = PolynomialFeatures(degree = 10)
X_modified_train = poly.fit_transform(X_train)   
X_modified_test = poly.fit_transform(X_test)

lasso_reg = Lasso()
lasso_reg.fit(X_modified_train,y_train)
lasso_train_rmse = math.sqrt(mean_squared_error(y_train,lasso_reg.predict(X_modified_train).reshape(-1,1)))
lasso_test_rmse = math.sqrt(mean_squared_error(y_test,lasso_reg.predict(X_modified_test).reshape(-1,1)))

lin_reg = LinearRegression()
lin_reg.fit(X_modified_train,y_train)
lin_train_rmse = math.sqrt(mean_squared_error(y_train,lin_reg.predict(X_modified_train).reshape(-1,1)))
lin_test_rmse = math.sqrt(mean_squared_error(y_test,lin_reg.predict(X_modified_test).reshape(-1,1)))

print('Lasso Regression - Training data RMSE ',lasso_train_rmse)
print('Lasso Regression - Test data RMSE ',lasso_test_rmse,'\n')
print('Linear Regression - Training data RMSE ',lin_train_rmse)
print('Linear Regression - Test data RMSE ',lin_test_rmse)


# ### Below code is to plot the various alpha values

# In[ ]:


train_err = []
test_err = []

alpha_vals = np.linspace(0,1,10)

for alpha_v in alpha_vals:
    reg = Lasso(alpha=alpha_v)
    reg.fit(X_modified_train,y_train)
    
    train_err.append(math.sqrt(mean_squared_error(y_train,reg.predict(X_modified_train).reshape(-1,1))))
    test_err.append(math.sqrt(mean_squared_error(y_test,reg.predict(X_modified_test).reshape(-1,1))))

print(train_err)
print(test_err)

plt.title('Lasso')
plt.xlabel('Alpha Degree')
plt.ylabel('RMSE')
plt.plot(np.linspace(0,1,10),train_err,'bo-',label='Lasso Train')
plt.plot(np.linspace(0,1,10),test_err,'ro-',label='Lasso Test')
plt.legend()
plt.show()


# In[ ]:




