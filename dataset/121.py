#!/usr/bin/env python
# coding: utf-8

# ## Polynomial Regression Model with sklearn and demonstration of Ridge Regression with default Alpha and Plot of RMSE for Alpha from 0.3 to 1
# 
# Completed code for the Project_Part D_#C1_Student ID No. 2018AIML632.  
# Project objective is to build a polynomial regression model for a given dataset and demonstrate Ridge regression. The regression models will be built by making use of libraries such as sklearn and numpy
# 
# ## Step 1 - Importing the required libraries  

# In[47]:


# Importing various required Libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math 

# the library we will use to create the model 
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import mean_squared_error


# ## Step 2 - Reading the dataset for Training and Testing separately

# In[48]:


# Reading the dataset from the project - part D - training data set.csv file using pandas &
#write into a pandas dataframe object named 'dataset'
dataset_train = pd.read_csv('project - part D - training data set.csv')
dataset_test = pd.read_csv('project - part D - testing data set.csv')

x_train = dataset_train['Father'].values.reshape(-1,1)
x_test = dataset_test['Father'].values.reshape(-1,1)
y_train = dataset_train ['Son'].values.reshape(-1,1)
y_test = dataset_test['Son'].values.reshape(-1,1)


# # Step3 - Performing Polynomial Regression for Degrees 1 to 10
# Using Scikit-Learn’s PolynomialFeatures class to transform the training data and testing data for polynomial of Degree 10. Modified_X now contains the original feature of X plus the polynomial features upto x power 10

# In[ ]:


poly_10=PolynomialFeatures(degree=10)
modified_x_train_10=poly_10.fit_transform(x_train)
modified_x_test_10=poly_10.fit_transform(x_test)


# # Step4a - Performing Ridge Regression with default Alpha

# In[86]:


reg_ridge=Ridge()
reg_ridge.fit(modified_x_train_10,y_train)
RMSE_train_ridge_er=math.sqrt(mean_squared_error(y_train,reg_ridge.predict(modified_x_train_10)))
RMSE_test_ridge_er=math.sqrt(mean_squared_error(y_test,reg_ridge.predict(modified_x_test_10)))
print('Ridge Reg. Train RMSE:',RMSE_train_ridge_er)
print('Ridge Reg. Test RMSE:',RMSE_test_ridge_er)
print('Ridge Regression Coefficient is :',reg_ridge.coef_)
print('Ridge Regression Intercept is :',reg_ridge.intercept_)


# # Step-4b Ridge Regression with Alpha varying from 0.3 to 1 and plot of RMSE Vs Alpha 
# (As Error at Alpha = 0 and 0.2 is very high, Ridge Regression done with Alpha from 0.3 to 1.)

# In[89]:


train_err_ridge=[]
test_err_ridge=[]

Poly_reg_10=PolynomialFeatures(degree=10)
modified_x_train_10=Poly_reg_10.fit_transform(x_train)
modified_x_test_10=Poly_reg_10.fit_transform(x_test)

print('Training Points:',modified_x_train_10.shape[0])
print('Testing Points:',modified_x_test_10.shape[0])

alpha_vals=np.linspace(0.3,1,10)
for alpha_v in alpha_vals:
    
    reg_ridge = Ridge(alpha=alpha_v, max_iter=1e4)
    reg_ridge.fit(modified_x_train_10,y_train)
    RMSE_train_er_ridge=math.sqrt(mean_squared_error(y_train,reg_ridge.predict(modified_x_train_10)))
    RMSE_test_er_ridge=math.sqrt(mean_squared_error(y_test,reg_ridge.predict(modified_x_test_10)))
    train_err_ridge.append(RMSE_train_er_ridge)
    test_err_ridge.append(RMSE_test_er_ridge)

plt.plot(np.linspace(0,1,10),train_err_ridge,'bo-',label='Train_Ridge_Reg')
plt.plot(np.linspace(0,1,10),test_err_ridge,'ro-',label='Test_Ridge_Reg')

plt.title("Ridge,Lasso,Lin Reg. Train/Test Er.Vs Alpha",fontsize=16,fontname="Arial",fontweight="bold")
plt.xlabel('Alpha',fontsize=14)
plt.ylabel('RMSE',fontsize=14)
plt.savefig('2018AIML632_Ridge_Vs_Lasso_Vs_Lin_Plot.png')
plt.legend()
plt.show()        


# In[ ]:




