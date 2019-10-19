#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn import linear_model
import numpy as np
import math
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from math import sqrt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


training_data = pd.read_csv('training.csv')
testing_data = pd.read_csv('testing.csv')
x_train = training_data['Father'].values.reshape(-1,1)
y_train = training_data['Son'].values.reshape(-1,1)
x_test = testing_data['Father'].values.reshape(-1,1)
y_test = testing_data['Son'].values.reshape(-1,1)


# In[4]:


polynomial_reg = PolynomialFeatures(degree=10)
x_modified_train = polynomial_reg.fit_transform(x_train)
x_modified_test = polynomial_reg.fit_transform(x_test)
model1 = linear_model.Lasso(alpha=0.5)
model1.fit(x_modified_train,y_train)
y_predicted_train = model1.predict(x_modified_train)
y_predicted_test = model1.predict(x_modified_test)
training_error = sqrt(mean_squared_error(y_train,y_predicted_train))
testing_error = sqrt(mean_squared_error(y_test,y_predicted_test))
print("Training Error with alpha value set to 0.5 : ",training_error)
print("Testing Error with alpha value set to 0.5 : ", testing_error)


# In[5]:


train_error=[]
test_error=[]

alpha_values = np.linspace(0,1,9)
for alpha_v in alpha_values:
    model2 = linear_model.Lasso(alpha=alpha_v)
    model2.fit(x_train,y_train)
    train_error.append(sqrt(mean_squared_error(y_train,model2.predict(x_train))))
    test_error.append(sqrt(mean_squared_error(y_test,model2.predict(x_test))))

print("test_errors", test_error)
plt.title("Lasso Regression")
plt.xlabel("Alpha Values")
plt.ylabel("RMSE")
plt.plot(np.linspace(0,1,9),train_error,'bo-',label="Training")
plt.plot(np.linspace(0,1,9),test_error,'ro-',label="Testing")
plt.legend()
plt.show()

