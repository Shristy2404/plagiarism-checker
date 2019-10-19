#!/usr/bin/env python
# coding: utf-8

# ## Polynomial Regression Model with sklearn and demonstration of Lasso Regression
# 
# Completed code for the Project_Part D_#C1_Student ID No. 2018AIML632.  
# Project objective is to build a polynomial regression model for a given dataset and demonstrate Lasso regression. The regression models will be built by making use of libraries such as sklearn and numpy
# 
# ## Step 1 - Importing the required libraries  

# In[1]:


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
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import mean_squared_error


# ## Step 2 - Reading the dataset for Training and Testing separately

# In[2]:


# Reading the dataset from the project - part D - training data set.csv file using pandas &
#write into a pandas dataframe object named 'dataset'
dataset_train = pd.read_csv('project - part D - training data set.csv')
dataset_test = pd.read_csv('project - part D - testing data set.csv')

x_train = dataset_train['Father'].values.reshape(-1,1)
x_test = dataset_test['Father'].values.reshape(-1,1)
y_train = dataset_train ['Son'].values.reshape(-1,1)
y_test = dataset_test['Son'].values.reshape(-1,1)


# # Step 3 - Performing Polynomial Regression for Degrees 1 to 10
# Using Scikit-Learn’s PolynomialFeatures class to transform the training data and testing data for polynomial of Degree 10. Modified_X now contains the original feature of X plus the polynomial features upto x power 10

# In[3]:


poly_lasso_10=PolynomialFeatures(degree=10)
modified_x_lasso_train=poly_lasso_10.fit_transform(x_train)
modified_x_lasso_test=poly_lasso_10.fit_transform(x_test)
reg_lasso=Lasso(alpha=1, max_iter=1e4)
reg_lasso.fit(modified_x_lasso_train,y_train)
RMSE_train_lasso_er=math.sqrt(mean_squared_error(y_train,reg_lasso.predict(modified_x_lasso_train)))
RMSE_test_lasso_er=math.sqrt(mean_squared_error(y_test,reg_lasso.predict(modified_x_lasso_test)))
print('Lasso Train RMSE:',RMSE_train_lasso_er)
print('Lasso Test RMSE:',RMSE_test_lasso_er)


# # Step 4 - Lasso Regression for Degrees 10 Polynomial

# In[4]:


train_err_lasso=[]
test_err_lasso=[]
train_err_lin=[]
test_err_lin=[]

Poly_reg_10=PolynomialFeatures(degree=10)
modified_x_train_10=Poly_reg_10.fit_transform(x_train)
modified_x_test_10=Poly_reg_10.fit_transform(x_test)

print('Training Points:',modified_x_train_10.shape[0])
print('Testing Points:',modified_x_test_10.shape[0])

alpha_vals=np.linspace(0,1,10)
for alpha_v in alpha_vals:
    
    reg_lasso = Lasso(alpha=alpha_v, max_iter=1e4)
    reg_lasso.fit(modified_x_train_10,y_train)
    RMSE_train_er_lasso=math.sqrt(mean_squared_error(y_train,reg_lasso.predict(modified_x_train_10)))
    RMSE_test_er_lasso=math.sqrt(mean_squared_error(y_test,reg_lasso.predict(modified_x_test_10)))
    train_err_lasso.append(RMSE_train_er_lasso)
    test_err_lasso.append(RMSE_test_er_lasso)
    
    reg_Linear = LinearRegression()
    reg_Linear.fit(modified_x_train_10,y_train)
    RMSE_train_er_lin=math.sqrt(mean_squared_error(y_train,reg_Linear.predict(modified_x_train_10)))
    RMSE_test_er_lin=math.sqrt(mean_squared_error(y_test,reg_Linear.predict(modified_x_test_10)))
    train_err_lin.append(RMSE_train_er_lin)
    test_err_lin.append(RMSE_test_er_lin)

plt.plot(np.linspace(0,1,10),train_err_lin,'go-',label='Train_Linear_Reg')
plt.plot(np.linspace(0,1,10),test_err_lin,'yo-',label='Test_Linear_Reg')    
plt.plot(np.linspace(0,1,10),train_err_lasso,'bo-',label='Train_Lasso_Reg')
plt.plot(np.linspace(0,1,10),test_err_lasso,'ro-',label='Test_Lasso_Reg')
plt.title("Lasso Vs Linear Reg.Train/Test Er. Vs Alpha",fontsize=16,fontname="Arial",fontweight="bold")
plt.xlabel('Alpha',fontsize=14)
plt.ylabel('RMSE',fontsize=14)
plt.savefig('2018AIML632_Lasso_Vs_Linear_Reg_Plot.png')
plt.legend()
plt.show()        


# In[ ]:





# In[ ]:





# In[ ]:




