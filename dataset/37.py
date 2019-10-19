#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# Remember what this line did?
get_ipython().run_line_magic('matplotlib', 'inline')
import math 
from math import sqrt

# the library we will use to create the model 
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso


# In[56]:


dataset = pd.read_csv('E:/BITS PILANI/project - part D - training data set.csv')
dataset1  = pd.read_csv('E:/BITS PILANI/project - part D - testing data set.csv')


# In[57]:


x_train = dataset['Father'].values.reshape(-1,1)
y_train = dataset['Son'].values.reshape(-1,1)
x_test  = dataset1['Father'].values.reshape(-1,1)
y_test =  dataset1['Son'].values.reshape(-1,1)


# In[58]:


trainerror=[]
testerror=[]


# In[59]:


polyreg = PolynomialFeatures(degree=10)
x_modified_train = polyreg.fit_transform(x_train)
x_modified_test = polyreg.fit_transform(x_test)
model = linear_model.Lasso(alpha=0.5)
model.fit(x_modified_train, y_train)
y_predicted_test=model.predict(x_modified_test)
y_predicted_train=model.predict(x_modified_train)
RMSE_train_data=sqrt(mean_squared_error(y_train,y_predicted_train))
RMSE_test_data=sqrt(mean_squared_error(y_test,y_predicted_test))
print("The RMSE for train data is \n",RMSE_train_data)
print("The RMSE for test data is \n",RMSE_test_data)


# In[60]:


alpha_values=np.linspace(0,1,9)
for alpha_value in alpha_values:
    polyreg=linear_model.Lasso(alpha=alpha_value)
    polyreg.fit(x_train, y_train)
    trainerror.append(sqrt(mean_squared_error(y_train,polyreg.predict(x_train))))
    testerror.append(sqrt(mean_squared_error(y_test,polyreg.predict(x_test))))


# In[ ]:





# In[ ]:





# In[ ]:




