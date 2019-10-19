
# coding: utf-8

# In[15]:




# In[4]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt # visualization library
# to plot the graph in the cell itself below statement
get_ipython().magic('matplotlib inline')
import math 

# the library we will use to create the model 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import metrics 

dataset=pd.read_csv('project - part D - training data set.csv')
x_train = dataset['Father']
x_train /= 100
x_train = x_train.values.reshape(-1,1)
y_train = dataset['Son'].values.reshape(-1,1)

dataset_test=pd.read_csv('project - part D - testing data set.csv')
x_test = dataset_test['Father']
x_test /= 100
x_test = x_test.values.reshape(-1,1)
y_test = dataset_test['Son'].values.reshape(-1,1)

degree_input=10
poly = PolynomialFeatures(degree=degree_input)
x_modified_train = poly.fit_transform(x_train)
x_modified_test = poly.fit_transform(x_test)

lasso = Lasso()
lasso.fit(x_modified_train,y_train)

print('Lasso train RMSE: {0} for Degree: {1}'.format(math.sqrt(mean_squared_error(y_train
                                    , lasso.predict(x_modified_train))), degree_input))

print('Lasso test RMSE: {0} for Degree: {1}'.format(math.sqrt(mean_squared_error(y_test
                                    , lasso.predict(x_modified_test))), degree_input))



# In[ ]:



