#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# Remember what this line did?
get_ipython().run_line_magic('matplotlib', 'inline')
import math 

# the library we will use to create the model 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics 


# In[11]:


df_train = pd.read_csv('project - part D - training data set.csv')
df_test = pd.read_csv('project - part D - testing data set.csv')
y_train = df_train['Son'].values.reshape(-1,1)
x_train = df_train['Father'].values.reshape(-1,1)
y_test = df_test['Son'].values.reshape(-1,1)
x_test = df_test['Father'].values.reshape(-1,1)
#print(x_test)


# In[12]:


poly = PolynomialFeatures(degree=10)
modified_x_train = poly.fit_transform(x_train)
modified_x_test = poly.fit_transform(x_test)

regr = Lasso()
regr.fit(modified_x_train,y_train)
train_rmse = math.sqrt(mean_squared_error(y_train,regr.predict(modified_x_train)))
test_rmse = math.sqrt(mean_squared_error(y_test,regr.predict(modified_x_test)))

#print(" Regularization(Lasso Regression) helps to counter overfitting. For Lasso regession with degree=10 RMSE is near to results otained by polynomial regression at degree=4 or RMSE of Lasso regression model (degree 10) < RMSE of polynomial model (degree 10)")
print("")
print(" Test RMSE (Lasso Regression at degree=%s) : %s" % (10,test_rmse))
print(" Train RMSE (Lasso Regression at degree=%s) : %s" % (10,train_rmse))  
    


# In[ ]:





# In[ ]:




