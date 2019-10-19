#!/usr/bin/env python
# coding: utf-8

# ## Lasso Regression
# 
# ## Step 1 - Importing the required libraries 
# 
# We have completed this step for you. Please go through it to have a clear idea about the libraries that are used in the project.

# In[44]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# Remember what this line did?
#%matplotlib inline  

# the library we will use to create the model 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso

from sklearn import metrics 
import math


# ## Step 2 - Reading the dataset and splitting it into testing and training data

# In[45]:


#Read the dataset from the csv file using pandas and write into a pandas dataframe object named 'dataset'
features = ['Father']
target = 'Son'
training_dataset = pd.read_csv('project - part D - training data set.csv')
testing_dataset = pd.read_csv('project - part D - testing data set.csv')


# In[46]:


#converting heights into meters for making it workable while performing a transpose
#xTrain = training_dataset[features].values.reshape((-1,1))
xTrain = training_dataset[features].values.reshape((-1,1))*(2.5/100)
yTrain = training_dataset[target].values.reshape((-1,1))
#xTest = testing_dataset[features].values.reshape((-1,1))
xTest = testing_dataset[features].values.reshape((-1,1))*(2.5/100)
yTest = testing_dataset[target].values.reshape((-1,1))


# ## Step 3 - Generating the model

# In[47]:


#for degree 10 plotting the graph for lambda vs rmse
d = 10
test_rmses = []
train_rmses = []
#transform xTrain for the current degree of polynomial
xTrainP = PolynomialFeatures(degree=d).fit_transform(xTrain)
alphas = np.asarray([x/10 for x in range(1,21)])
for alpha in alphas:
    #train model
    lassoRegressor = Lasso(alpha=alpha)
    lassoRegressor.fit(xTrainP, yTrain)
    #validate the model with test dataset
    xTestP = PolynomialFeatures(degree=d).fit_transform(xTest)
    #calculate rmse
    train_rmse = math.sqrt(metrics.mean_squared_error(yTrain,lassoRegressor.predict(xTrainP)))
    train_rmses.append(train_rmse)
    test_rmse = math.sqrt(metrics.mean_squared_error(yTest,lassoRegressor.predict(xTestP)))
    test_rmses.append(test_rmse)
    print("Train RMSE Value for {0} degree @ Lamda {1} is : {2}".format(d, alpha,train_rmse))
    print("Test RMSE Value for {0} degree @ Lamda {1} is : {2}".format(d, alpha, test_rmse))
    #print("Coefficients : {0}".format(lassoRegressor.coef_))

xaxis = alphas
plt.xlabel("Lamda")
plt.ylabel("RMSE")
plt.plot(xaxis,train_rmses,'bo-',label="Train")
plt.plot(xaxis,test_rmses,'ro-',label="Test")
plt.legend()
plt.savefig('2018aiml621_lasso_lambda_part_d.png')
plt.show()

test_rmses = []
train_rmses = []
#for each degree from 1 to 10 plotting the graph for polynomial degree vs rmse
for d in range(1,11):
    #transform xTrain for the current degree of polynomial
    xTrainP = PolynomialFeatures(degree=d).fit_transform(xTrain)
    #train model
    lassoRegressor = Lasso()
    lassoRegressor.fit(xTrainP, yTrain)

    #validate the model with test dataset
    xTestP = PolynomialFeatures(degree=d).fit_transform(xTest)
    
    #calculate rmse
    train_rmse = math.sqrt(metrics.mean_squared_error(yTrain,lassoRegressor.predict(xTrainP)))
    train_rmses.append(train_rmse)
    test_rmse = math.sqrt(metrics.mean_squared_error(yTest,lassoRegressor.predict(xTestP)))
    test_rmses.append(test_rmse)
    print("Train RMSE Value for {0} degree is : {1}".format(d, train_rmse))
    print("Test RMSE Value for {0} degree is : {1}".format(d, test_rmse))
    #print("Coefficients : {0}".format(lassoRegressor.coef_))
    
xaxis = np.linspace(1,10,10)
plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")
plt.plot(xaxis,train_rmses,'bo-',label="Train")
plt.plot(xaxis,test_rmses,'ro-',label="Test")
plt.legend()
plt.savefig('2018aiml621_lasso_degree_part_d.png')
plt.show()


# In[ ]:





# In[ ]:




