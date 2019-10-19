#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# Remember what this line did?
get_ipython().run_line_magic('matplotlib', 'inline')
import math 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# the library we will use to create the model 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

#from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score
import operator

import warnings
warnings.filterwarnings("ignore")


# In[60]:


# 2.1 Read the dataset from the csv file using pandas and write into a pandas dataframe object named 'dataset'
train_dataset=pd.read_csv('proj_part_D_train.csv')
test_dataset=pd.read_csv('proj_part_D_test.csv')


# In[61]:


#x = train_dataset['Father'].values.reshape(-1,1)
#y = train_dataset['Son'].values.reshape(-1,1)

x_train = train_dataset['Father'].values.reshape(-1,1)
y_train = train_dataset['Son'].values.reshape(-1,1)

x_test = test_dataset['Father'].values.reshape(-1,1)
y_test = test_dataset['Son'].values.reshape(-1,1)

print(train_dataset.shape)
print(x_train.shape)
train_dataset.head()

print(test_dataset.shape)
print(x_test.shape)
test_dataset.head()


# In[62]:


# Lasso Regression ----- polynomial degree 10 and default alpha value
   
poly_10=PolynomialFeatures(degree=10)
modified_x_train=poly_10.fit_transform(x_train)
modified_x_test=poly_10.fit_transform(x_test)

#reg_lasso=Lasso(fit_intercept=False,tol=0.01,max_iter=500000)
reg_lasso=Lasso(fit_intercept=False,tol=0.001,max_iter=100000)
# Build Model for Train Data using LASSO regression
reg_lasso.fit(modified_x_train,y_train)
Yp_train=reg_lasso.predict(modified_x_train)
RMSE_train=math.sqrt(mean_squared_error(y_train,Yp_train))
# Predict using builtup model on test data
Yp_test=reg_lasso.predict(modified_x_test)
RMSE_test=math.sqrt(mean_squared_error(y_test,Yp_test))
train_err.append(RMSE_train)
test_err.append(RMSE_test)
print("Train Err: ", RMSE_train," Test Err: ", RMSE_test )
print("")
print("Model Parametrs Obtained as:")
print("Coefficient: ", reg_lasso.coef_)
print("Intercept: ", reg_lasso.intercept_)


# In[63]:


##### COMAPRISON OF NORMAL POLYNOMIAL, RIDGE AND LASSO REGRESSION WITH RESPECT TO RMSE #####
#------------------------------------------------------------------------------------------#
i=0
train_err=[]
test_err=[]
train_err_ridge=[]
test_err_ridge=[]
train_err_lasso=[]
test_err_lasso=[]
for i in range(1,11):
    poly=PolynomialFeatures(degree=i)
    modified_x_train=poly.fit_transform(x_train)
    modified_x_test=poly.fit_transform(x_test)
    reg=LinearRegression()
    # Build Model for Train Data
    reg.fit(modified_x_train,y_train)
    Yp_train=reg.predict(modified_x_train)
    RMSE_train=math.sqrt(mean_squared_error(y_train,Yp_train))
    # Predict using builtup model on test data
    Yp_test=reg.predict(modified_x_test)
    RMSE_test=math.sqrt(mean_squared_error(y_test,Yp_test))
    train_err.append(RMSE_train)
    test_err.append(RMSE_test)
    #### RIDGE Model
    reg_ridge=Ridge()
    reg_ridge.fit(modified_x_train,y_train)
    Yp_train_ridge=reg_ridge.predict(modified_x_train)
    RMSE_train_ridge=math.sqrt(mean_squared_error(y_train,Yp_train_ridge))
    Yp_test_ridge=reg_ridge.predict(modified_x_test)
    RMSE_test_ridge=math.sqrt(mean_squared_error(y_test,Yp_test_ridge))
    train_err_ridge.append(RMSE_train_ridge)
    test_err_ridge.append(RMSE_test_ridge)
    #### LASSO Model
    reg_lasso=Lasso(max_iter=100000)
    reg_lasso.fit(modified_x_train,y_train)
    Yp_train_lasso=reg_lasso.predict(modified_x_train)
    RMSE_train_lasso=math.sqrt(mean_squared_error(y_train,Yp_train_lasso))
    Yp_test_lasso=reg_lasso.predict(modified_x_test)
    RMSE_test_lasso=math.sqrt(mean_squared_error(y_test,Yp_test_lasso))
    train_err_lasso.append(RMSE_train_lasso)
    test_err_lasso.append(RMSE_test_lasso)
    
plt.title('Polynomial vs Ridge vs Lasso' )
plt.xlabel('Polynomial Degree')
plt.ylabel('RMSE')
plt.plot(np.linspace(1,11,10),train_err,'bo-',label='Poly_Train')
plt.plot(np.linspace(1,11,10),test_err,'ro-',label='poly_Test')
plt.plot(np.linspace(1,11,10),train_err_ridge,'go-',label='Ridge_Train')
plt.plot(np.linspace(1,11,10),test_err_ridge,'mo-',label='Ridge_Test')
plt.plot(np.linspace(1,11,10),train_err_lasso,'co-',label='Lasso_Train')
plt.plot(np.linspace(1,11,10),test_err_lasso,'yo-',label='Lasso_Test')
plt.legend()
plt.show()
plt.close()


# In[64]:


# Typical Regression Model of Polynomial degree 10
poly=PolynomialFeatures(degree=10)
modified_x_train=poly.fit_transform(x_train)
modified_x_test=poly.fit_transform(x_test)
#print(modified_x_test)
poly.fit(modified_x_train,y_train)
reg=LinearRegression()
# Build Model for Train Data
reg.fit(modified_x_train,y_train)
Yp_train=reg.predict(modified_x_train)
RMSE_train=math.sqrt(mean_squared_error(y_train,Yp_train))
# Predict using builtup model on test data
Yp_test=reg.predict(modified_x_test)
RMSE_test=math.sqrt(mean_squared_error(y_test,Yp_test))
    
print(" Train Err: ", RMSE_train," Test Err: ", RMSE_test )
print("")
print("Model Parametrs Obtained as:")
print("Coefficient: ", reg.coef_)
print("Intercept: ", reg.intercept_)


# In[65]:


####### PLOT POLYNOMIAL CURVE AGAINST INPUT DATASET ####
# PLOT 
plt.scatter(x_test, y_test,color='blue',label='Test')
plt.scatter(x_train, y_train,color='red',label='Train')
plt.xlabel('Fathers Height')
plt.ylabel('Sons Height')
#Plot Test
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x_test,Yp_test), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='blue',label='Test,Poly=10')
#Plot Train
sorted_zip = sorted(zip(x_train,Yp_train), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='red',label='Train,Poly=10')
#------------------ PLOT FOR LASSO POLYNOMIAL CURVE -------------------
sorted_zip = sorted(zip(x_train,reg_lasso.predict(modified_x_train)), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='green',label='Lasso')
plt.legend()
plt.show()


# In[67]:


# Lasso Regression ---- polynomial degree 10 and varying alpha value

i=0
train_err=[]
test_err=[]
    
poly_10=PolynomialFeatures(degree=10)
modified_x_train=poly_10.fit_transform(x_train)
modified_x_test=poly_10.fit_transform(x_test)

alpha_vals=np.linspace(0.01,1,10)
for a_val in alpha_vals:
    reg=Lasso(alpha=a_val,fit_intercept=False,tol=0.01,max_iter=1000)
    # Build Model for Train Data using LASSO regression
    reg.fit(modified_x_train,y_train)
    Yp_train=reg.predict(modified_x_train)
    RMSE_train=math.sqrt(mean_squared_error(y_train,Yp_train))
    # Predict using builtup model on test data
    Yp_test=reg.predict(modified_x_test)
    RMSE_test=math.sqrt(mean_squared_error(y_test,Yp_test))
    train_err.append(RMSE_train)
    test_err.append(RMSE_test)
    #print(i, " For Alpha: ", a_val, "\n Train Err: ", RMSE_train," Test Err: ", RMSE_test )
    i=i+1

###### PLOT GRAPH FOR LASSO #######
plt.title('Lasso Regression' )
plt.xlabel('Alpha Values' )
plt.ylabel('RMSE')
plt.plot(np.linspace(0,1,10),train_err,'bo-',label='Train')
plt.plot(np.linspace(0,1,10),test_err,'ro-',label='Test')
plt.legend()
plt.show()
#plt.savefig('2018AIML638_Poly_Part_D.png')
plt.close()


# In[ ]:





# In[ ]:




