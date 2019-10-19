#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import section
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math


# In[16]:


trainning_dataset = pd.read_csv('project - part D - training data set.csv')
X_train= trainning_dataset['Father'].values.reshape(-1, 1)
Y_train= trainning_dataset['Son'].values.reshape(-1, 1)

testing_dataset = pd.read_csv('project - part D - testing data set.csv')
X_test= testing_dataset['Father'].values.reshape(-1, 1)
Y_test= testing_dataset['Son'].values.reshape(-1, 1)

poly = PolynomialFeatures(degree=10)
x_train_transformed = poly.fit_transform(X_train)
x_test_transformed  = poly.fit_transform(X_test)

lassoPoly = Lasso(alpha = 0.5)
lassoPoly.fit(x_train_transformed, Y_train)


# In[17]:


train_error = []
test_error  = []

alpha = np.linspace(0,1,10)
for val in alpha:
    lassoreg= Lasso(alpha=val)
    lassoreg.fit(X_train,Y_train)
    train_error.append(math.sqrt(mean_squared_error(Y_train, lassoreg.predict(X_train))))
    test_error.append(math.sqrt(mean_squared_error(Y_test, lassoreg.predict(X_test))))
    
print("RMSE for trainning data ",math.sqrt(mean_squared_error(Y_train, lassoPoly.predict(x_train_transformed))))
print("RMSE for testing data "  ,math.sqrt(mean_squared_error(Y_test, lassoPoly.predict(x_test_transformed))))
plt.title('**** Lasso ****')
plt.xlabel('Alpha value')
plt.ylabel('RMSE')

plt.plot(np.linspace(0,1,10),train_error,'bo-',label="Train")
plt.plot(np.linspace(0,1,10),test_error,'ro-',label="Test")

plt.legend()
plt.show()


# In[ ]:




