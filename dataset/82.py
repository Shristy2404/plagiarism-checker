#!/usr/bin/env python
# coding: utf-8

# # Building a Polynomial Regression model with Lasso Regularisation method

# In[20]:


import numpy as np
import pandas as pd
from sklearn import linear_model as lm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


training_data = pd.read_csv("project - part D - training data set.csv")
testing_data = pd.read_csv("project - part D - testing data set.csv")


# In[22]:


x_train = training_data[['Father']]
y_train = training_data[['Son']].values.reshape(-1,1)
x_test = testing_data[['Father']]
y_test = testing_data['Son'].values.reshape(-1,1)


# In[23]:


from sklearn.preprocessing import PolynomialFeatures as pf
model = lm.Lasso() #Default alpha = 1.0 is used when alpha value is not explicitly mentioned

poly = pf(degree=10)
modified_x_train = poly.fit_transform(x_train)
model.fit(modified_x_train,y_train)
y_pred = model.predict(modified_x_train)
y_pred = y_pred.reshape(-1,1) 

#Metrics on the Accuracy of the model with training data
MSE = (1/len(y_train))*np.sum((y_pred - y_train)**2)
RMSE = np.sqrt(MSE)

print("Training Root Mean Squared Error (RMSE): ", RMSE)

modified_x_test = poly.fit_transform(x_test)
y_pred = model.predict(modified_x_test)
y_pred = y_pred.reshape(-1,1)

#Metrics on the Accuracy of the model with test data
MSE = (1/len(y_test))*np.sum((y_pred - y_test)**2)
RMSE = np.sqrt(MSE)

print("Testing Root Mean Squared Error (RMSE): ", RMSE)


# ### Training and Testing RMSE values with PolynomialRegression method are 1.3787 and 1.8179 respectively. Comparing these with the ones obtained in LassoRegression above, we can infer that testing error is reduced a bit and training error is increased slightly in Lasso.
# 
# ### Also as shown in the warning message above, we did not yet achieve the convergence for the default alpha value of 1.0 and hence the model did not improve much over the model obtained in PolynomialRegression. So, we might get a better model as we tune alpha value with good no. of iterations which is out of scope for this assignment.
