#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import PolynomialFeatures

#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
#from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import math


# In[38]:


traindata = pd.read_csv('train_data.csv')
testdata  = pd.read_csv('test_data.csv')
x_train = traindata['Father'].values.reshape(-1,1)
y_train = traindata['Son'].values.reshape(-1,1)

x_test = testdata['Father'].values.reshape(-1,1)
y_test = testdata['Son'].values.reshape(-1,1)

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)


# In[39]:


poly_reg = PolynomialFeatures(degree=11)
x_modified_train = poly_reg.fit_transform(x_train)
x_modified_test  = poly_reg.fit_transform(x_test)

print(x_modified_train.shape)
print(x_modified_test.shape)


# In[41]:


reg = Lasso(alpha=1.0)
reg.fit(x_modified_train,y_train)

y_predicted_train = reg.predict(x_modified_train)
y_predicted_test  = reg.predict(x_modified_test)

tr_err = math.sqrt(mean_squared_error(y_train,y_predicted_train))
tt_err = math.sqrt(mean_squared_error(y_test,y_predicted_test))

print('Lasso Train RMSE :', tr_err)
print('Lasso Test  RMSE :', tt_err)


# In[ ]:


#RMSE values For Polynomial of Degree 10 
#Ridge Train RMSE      : 1.374786046795676
#Ridge Test  RMSE      : 2.0212244431165325
#Lasso Train RMSE      : 1.4422673630444791
#Lasso Test  RMSE      : 1.5408335797497317
#Polynomial TRAIN RMSE : 1.8179720763695844
#Polynomial TEST RMSE  : 1.3787128183638069

#For given data set, The RIDGE method fits the data for training.
#For Test data LASSO predicts more correctly.

