#!/usr/bin/env python
# coding: utf-8

# In[92]:


# Step 1 - Importing the required libraries

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
import math 

# the library we will use to create the model 

from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn import metrics 


# In[93]:


# Read the dataset from the csv file using pandas and write into a pandas dataframe object named 'dataset'

train_data = pd.read_csv("project - part D - training data set.csv")
print (train_data)
test_data = pd.read_csv("project - part D - testing data set.csv")
print (test_data)


# In[94]:


# Read test and train data. 

x = train_data['Father'].values.reshape(-1,1)
y = train_data['Son'].values.reshape(-1,1)

print (x)
print(y)

x_test = test_data['Father'].values.reshape(-1,1)
y_test = test_data['Son'].values.reshape(-1,1)

print (x_test)
print(y_test)

x_train = x
x_test_m = x_test

poly = PolynomialFeatures(degree=11)
x_train = poly.fit_transform(x)
x_test_m = poly.fit_transform(x_test)
     
print('modified xtrain', x_train)
print('modified xtest', x_test_m)

scaler = StandardScaler()
    
scaler.fit(x_train)                        #  Fit only on training data
x_train = scaler.transform(x_train)
       
scaler.fit(x_test_m)    
x_test_m = scaler.transform(x_test_m)     # apply same transformation to test data


# In[95]:


# Generate a model object using the library imported above 

test_mse = np.zeros((10,1))
test_rmse = np.zeros((10,1))

train_mse = np.zeros((10,1))
train_rmse = np.zeros((10,1))

alpha_vals = np.linspace(0.1,1,10)

j=0

for i in alpha_vals:

    reg_model = Lasso(alpha=i)
    
    reg_model.fit(x_train, y)
    
    y_pred_test = reg_model.predict(x_test_m)
    
    test_mse[j] = mean_squared_error(y_test, y_pred_test)
    test_rmse[j] = math.sqrt(test_mse[j])
    print ('test rmse', test_rmse[j])
    
    y_pred_train = reg_model.predict(x_train) 
    
    train_mse[j] = mean_squared_error(y, y_pred_train)
    train_rmse[j] = math.sqrt(train_mse[j])
    print ('train RMSE', train_rmse[j])
         
    j += 1

print ('Test RMSE', test_rmse)
print ('Train RMSE', train_rmse)


# In[96]:


# Visualizing the line obtained 

fig1, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis

plt.plot(alpha_vals,test_rmse, color='r', linewidth=1,label="Test")
plt.plot(alpha_vals,test_rmse,'bo', color='r')
plt.plot(alpha_vals,train_rmse, color='blue', linewidth=1,label="Train")
plt.plot(alpha_vals,train_rmse,'bo', color='blue')
plt.title('Lasso')
plt.xlabel("Alpha" )
plt.ylabel("RMSE")
plt.legend()
plt.show()
fig1.savefig('Lasso_rmse.png') 
plt.close(fig1) 


# In[ ]:




