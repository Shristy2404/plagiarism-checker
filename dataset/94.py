#!/usr/bin/env python
# coding: utf-8

# ## Lasso Regression

# In[7]:


#Importing required libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math


# In[8]:


# Reading the dataset from the csv file using pandas and writing it into a pandas dataframe object named 'dataset'
df_train = pd.read_csv('project - part D - training data set.csv')
df_test = pd.read_csv('project - part D - testing data set.csv')


# In[9]:


x_train = df_train['Father'].values.reshape(-1,1)
y_train = df_train['Son'].values.reshape(-1,1)
x_test = df_test['Father'].values.reshape(-1,1)
y_test = df_test['Son'].values.reshape(-1,1)


# In[10]:


#Building the Lasso Regression model
poly = PolynomialFeatures(degree=10)
modified_x_train = poly.fit_transform(x_train)
modified_x_test = poly.fit_transform(x_test)
reg = Lasso()
reg.fit(modified_x_train,y_train)


# In[11]:


#Evaluation of Lasso Regression model
Lasso_Train_RMSE = math.sqrt(mean_squared_error(y_train,reg.predict(modified_x_train)))
Lasso_Test_RMSE = math.sqrt(mean_squared_error(y_test,reg.predict(modified_x_test)))
print('Lasso train_RMSE:',Lasso_Train_RMSE)
print('Lasso test_RMSE:',Lasso_Test_RMSE)


# In[12]:


#Building Polynomial Regression model for comparison
reg = LinearRegression().fit(modified_x_train,y_train)
Poly_Train_RMSE = math.sqrt(mean_squared_error(y_train,reg.predict(modified_x_train)))
Poly_Test_RMSE = math.sqrt(mean_squared_error(y_test,reg.predict(modified_x_test)))

print('Poly train_RMSE:',Poly_Train_RMSE)
print('Poly test_RMSE:',Poly_Test_RMSE)


# In[14]:


n_groups = 2
Poly = [Poly_Train_RMSE,Poly_Test_RMSE]
Lasso = [Lasso_Train_RMSE,Lasso_Test_RMSE]

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.3

rects1 = plt.bar(index, Poly, bar_width, color='b', label='Polynomial Regression')
rects1 = plt.bar(index+bar_width, Lasso, bar_width, color='g', label='Lasso Regression')

plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('Test and Train RMSE by Model')
plt.xticks(index + bar_width, ('Train','Test'))
plt.tight_layout()
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)

print("Although Training RMSE may have increased, Test RMSE is reduced implementing Lasso regression compared to Polynomial Regression as shown in the plot below. \n \n Lasso regression reduces the error by constraining the degrees of freedom of the model parameters. \n \n Test Error is reduced from 1.8180 to 1.5418.")

