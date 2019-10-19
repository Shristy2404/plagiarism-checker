#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing the required libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math 
import statistics

from sklearn.preprocessing import PolynomialFeatures
# the library we will use to create the model 
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics 
from sklearn.metrics import mean_squared_error

#Reading the data file and writing the data into pandas dataframe object: 'dataset_train'
dataset_train = pd.read_csv('project - part D - training data set.csv')

#Reshaping the data
x_train = dataset_train['Father'].values.reshape(-1,1)
y_train = dataset_train['Son'].values.reshape(-1,1)

#Reading the data file and writing the data into pandas dataframe object: 'dataset_train'
dataset_test = pd.read_csv('project - part D - testing data set.csv')

#Reshaping the test data
x_test = dataset_test['Father'].values.reshape(-1,1)
y_test = dataset_test['Son'].values.reshape(-1,1)

#Lasso Regression with Alpha as default value
#Polynomial regression
polynomial = PolynomialFeatures(degree=10)
mod_x_train = polynomial.fit_transform(x_train)
mod_x_test = polynomial.fit_transform(x_test)

#Linear regression model fitting
reg = LinearRegression() 
res = reg.fit(mod_x_train,y_train)

#Calculating Performance metrics on Training Data and printing them
y_pred_train = reg.predict(mod_x_train)
mse = mean_squared_error(y_train,y_pred_train)
rmse= math.sqrt(mse)
print('Polynomial Regression of Degree 10: Train RMSE:',rmse)

#Calculating Performance metrics on Testing Data and printing them
y_pred_test = reg.predict(mod_x_test)
mse_test = mean_squared_error(y_test,y_pred_test)
rmse_test= math.sqrt(mse_test)
print('Polynomial Regression of Degree 10: Test RMSE:',rmse_test)

#Lasso Regression
lass = Lasso() 
lass_res = lass.fit(mod_x_train,y_train)

print('Alpha value is:', lass.alpha)
print('Max Iterations value is:', lass.max_iter)
print('tol value is:', lass.tol)

#Calculating Performance metrics on Training Data and printing them
y_pred_train = lass.predict(mod_x_train)
mse = mean_squared_error(y_train,y_pred_train)
rmse_lasso= math.sqrt(mse)
print('Lasso Train RMSE:',rmse_lasso)

#Calculating Performance metrics on Testing Data and printing them
y_pred_test = lass.predict(mod_x_test)
mse_test = mean_squared_error(y_test,y_pred_test)
lasso_rmse_test= math.sqrt(mse_test)
print('Lasso Test RMSE:',lasso_rmse_test)


# In[3]:


#Lasso Regression with Alpha values between 0.1 to 1
#Polynomial regression
polynomial = PolynomialFeatures(degree=10)
mod_x_train = polynomial.fit_transform(x_train)
mod_x_test = polynomial.fit_transform(x_test)

#Linear regression model fitting
reg = LinearRegression() 
res = reg.fit(mod_x_train,y_train)

#Calculating Performance metrics on Training Data and printing them
y_pred_train = reg.predict(mod_x_train)
mse = mean_squared_error(y_train,y_pred_train)
rmse= math.sqrt(mse)
print('Train RMSE:',rmse)

#Calculating Performance metrics on Testing Data and printing them
y_pred_test = reg.predict(mod_x_test)
mse_test = mean_squared_error(y_test,y_pred_test)
rmse_test= math.sqrt(mse_test)
print('Test RMSE:',rmse_test)

#Ridge Regression
lasso_train_error = []
lasso_test_error = []
#Ridge regression model fitting
alpha_values = np.linspace(0.1,1,10)
for alpha_value in alpha_values:
    lass = Lasso(alpha = alpha_value) 
    lass_res = lass.fit(mod_x_train,y_train)
    print('Alpha value is:', lass.alpha)
    #Calculating Performance metrics on Training Data and printing them
    y_pred_train = lass.predict(mod_x_train)
    mse = mean_squared_error(y_train,y_pred_train)
    rmse_lasso= math.sqrt(mse)
    lasso_train_error.append(rmse_lasso)
    print('Lasso Train RMSE:',rmse_lasso)
    
    #Calculating Performance metrics on Testing Data and printing them
    y_pred_test = lass.predict(mod_x_test)
    mse_test = mean_squared_error(y_test,y_pred_test)
    lasso_rmse_test= math.sqrt(mse_test)
    lasso_test_error.append(lasso_rmse_test)
    print('Lasso Test RMSE:',lasso_rmse_test)
    
print('The minimum Test RMSE obtained is:',min(lasso_test_error),'for the Alpha value:',
      alpha_values[lasso_test_error.index(min(lasso_test_error))])
    
plt.title('Lasso Regression with different Alpha values')
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.plot(alpha_values,lasso_train_error,'bo-',label='Train')
plt.plot(alpha_values,lasso_test_error,'ro-',label='Test')
plt.legend()
plt.savefig('2018aiml533_lasso_part_d.png')

