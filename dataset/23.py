#!/usr/bin/env python
# coding: utf-8

# ## Polynomial Linear Regression with Normal Equations
#   
# 
# ## Step 1 - Importing the required libraries 
# 

# In[8]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
import math 

# the library we will use to create the model 
#from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn import linear_model
#from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error,mean_absolute_error
from math import sqrt


# ## Step 2 - Reading the dataset and splitting it into testing and training data

# In[9]:


# 2.1 Read the Training dataset from the csv file using pandas and write into a pandas dataframe object named 'dsTrain'
dsTrain = pd.read_csv("project - part D - training data set.csv")


# Now we need to process the data so that the sklearn function for generating the model can be invoked. This involves converting the data into numpy arrays standardizing it.

# In[10]:


xTrain = dsTrain['Father'].values.reshape(-1,1)
yTrain = dsTrain['Son'].values.reshape(-1,1)


# In[11]:


# 2.1 Read the Training dataset from the csv file using pandas and write into a pandas dataframe object named 'dsTest'
dsTest = pd.read_csv("project - part D - testing data set.csv")
xTest  = dsTest['Father'].values.reshape(-1,1)
yTest  = dsTest['Son'].values.reshape(-1,1)


# ## Step 3 - Building model and Performance measures
# 
#  

# In[44]:


poly = preprocessing.PolynomialFeatures(degree=10)
modified_xTrain = poly.fit_transform(xTrain)
modified_xTest = poly.fit_transform(xTest)
reg = linear_model.Lasso(max_iter=60000)
reg.fit(modified_xTrain, yTrain)

print('Lasso Train RMSE: ',math.sqrt(mean_squared_error(yTrain, reg.predict(modified_xTrain))))
print('Lasso Test  RMSE: ',math.sqrt(mean_squared_error(yTest, reg.predict(modified_xTest))))
print('Lasso coef', reg.coef_)
print('Lasso intercept', reg.intercept_)


# In[45]:


train_err = []
test_err = []
least_test_err = 0
poly_10 = preprocessing.PolynomialFeatures(degree=10)
modified_xTrain = poly_10.fit_transform(xTrain)
modified_xTest = poly_10.fit_transform(xTest)

alphas_vals = np.linspace(0.1,1,10)
for alpha_v in alphas_vals:
    reg = linear_model.Lasso(alpha=alpha_v, max_iter =800000)
    reg.fit(modified_xTrain, yTrain)
    
    test_rmse = math.sqrt(mean_squared_error(yTest, reg.predict(modified_xTest)))
    if (least_test_err == 0):
        least_test_err = test_rmse
        best_alpha = alpha_v
    else:
        if(least_test_err > test_rmse):
            least_test_err = test_rmse
            best_alpha = alpha_v
    train_err.append(math.sqrt(mean_squared_error(yTrain, reg.predict(modified_xTrain))))
    test_err.append(test_rmse)
    print(test_rmse)
    
plt.title('Lasso')
plt.xlabel('Alpha value')
plt.ylabel('RMSE')
plt.plot(np.linspace(0.1,1,10),train_err,'bo-',label='Train')
plt.plot(np.linspace(0.1,1,10),test_err,'ro-',label='Test')
plt.legend()
plt.show()

print('Best Alpha : ', best_alpha)

    


# In[ ]:





# In[ ]:




