#!/usr/bin/env python
# coding: utf-8

# ## Polynomial Regression
# 
# ## Step 1 - Importing the required libraries 
# 
# Here are the libraries that will used in the project.

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as py
import math 
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures 


# ## Step 2 - Reading the dataset for testing and training data

# In[2]:


# Read the dataset from the csv file using pandas and write into a pandas dataframe object named 'df'
df_train = pd.read_csv("project - part D - training data set.csv")
df_train.shape
df_train.head()


# In[3]:


df_test = pd.read_csv("project - part D - testing data set.csv")
df_test.shape
df_test.head()


# In[4]:


X_train = df_train['Father'].values.reshape(-1,1)
Y_train = df_train['Son'].values.reshape(-1,1)

X_test = df_test['Father'].values.reshape(-1,1)
Y_test = df_test['Son'].values.reshape(-1,1)


print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
# print(X_train, Y_train)
# print(X_test, Y_test)


# In[5]:


# Converting the data set into polynomial, which will be represented as multiple liner regression
# where for 3 degree polynomial, x1 = x, x2 = x^2, x3 = x^3

poly = PolynomialFeatures(degree=10)
mod_X_train = poly.fit_transform(X_train)
mod_X_test = poly.fit_transform(X_test)

print(mod_X_train.shape, Y_train.shape)
print(mod_X_test.shape, Y_test.shape)


# ## Step 3 - Generating the model 
# 
# Now that we have prepared the data, we proceed to generate a model for the same. 
# Here we need to make use of the Lasso library we imported above and then 'fit' the training data to the model object. 

# In[6]:


# 2.4 generate a model object using the library imported above 

Reg = Lasso(alpha=1,max_iter=100000,tol=0.9)
Reg.fit(mod_X_train, Y_train)
# Reg.fit(np.concatenate((mod_X_train,mod_X_test)), np.concatenate((Y_train,Y_test)))


# ## Step 4 - Calculating the Evaluation Measures
# 
# Here we will find mean square error for Train and Test data.
# 

# In[7]:


# Evaluating the measures:

train_err = math.sqrt(mean_squared_error(Y_train,Reg.predict(mod_X_train)))
test_err = math.sqrt(mean_squared_error(Y_test,Reg.predict(mod_X_test)))

print("Train RMSE score is: ",train_err)
print("Test RMSE score is: ",test_err)


# ## Step 5 - Creating the same model and evaluate the error for ploynomials of degree 10
# 
# We will use polynomials of degree 10 and try to find the mean square error on Train and Test data, for varing values of alpha between 0 and 1.

# In[23]:


train_err = []
test_err = []

alpha_vals = np.linspace(0,1,10)

for i in alpha_vals:
    reg = Lasso(alpha=i,max_iter=500000,tol=0.9)
    reg.fit(np.concatenate((mod_X_train,mod_X_test)), np.concatenate((Y_train,Y_test)))
    
    train_err.append(math.sqrt(mean_squared_error(Y_train,reg.predict(mod_X_train))))
    test_err.append(math.sqrt(mean_squared_error(Y_test,reg.predict(mod_X_test))))
    
print(train_err)
print(test_err)


# ## Step 6 - Visualing the test and train error for ploynomials of degree 10 with varing alpha values

# In[24]:


py.title("Ridge")
py.xlabel("Alpha Value")
py.ylabel("Error - RMSE value")
py.plot(np.linspace(0,1,10),train_err,'bo-',label='Train')
py.plot(np.linspace(0,1,10),test_err,'ro-',label='Test')
py.legend()
py.show()

print('With the given data set and polynomial of degree 10, the Lasso Regression model does not fit good for any value of alpha between 0 and 1.')

