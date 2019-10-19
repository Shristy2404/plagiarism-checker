#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error


# In[3]:


# Read Data
train_df = pd.read_csv("project - part D - training data set.csv")
test_df = pd.read_csv("project - part D - testing data set.csv")


# In[4]:


X_train, Y_train = train_df["Father"].values.reshape(-1,1), train_df["Son"].values.reshape(-1,1)
X_test, Y_test = test_df["Father"].values.reshape(-1,1), test_df["Son"].values.reshape(-1,1)


# In[5]:


def get_rmse(X, Y, reg):
    Y_pred = reg.predict(X)
    rse = mean_squared_error(Y, Y_pred)
    return math.sqrt(rse)

def plot(vals, train_array, test_array):
    plt.plot(vals, train_array, '--o--', color="blue", label="Training")
    plt.plot(vals, test_array, '--o--', color="red", label="Test")
    plt.xlabel("alpha")
    plt.ylabel("E-RMSE")
    plt.legend()
    plt.show()


# In[6]:



deg = 10

poly = PolynomialFeatures(degree=deg,include_bias=False)
modified_X_train = poly.fit_transform(X_train)
lasso_reg = Lasso()
lasso_reg.fit(modified_X_train, Y_train)

rmse_train = get_rmse(modified_X_train, Y_train, lasso_reg)

modified_X_test = poly.fit_transform(X_test)
rmse_test = get_rmse(modified_X_test, Y_test, lasso_reg)

print ("Lasso with Deg: 10 - RMSE Train: {} RMSE Test {}".format(rmse_train, rmse_test))
# plot(iters, rmse_train_array, rmse_test_array)