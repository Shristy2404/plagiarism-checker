
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math 

from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing


# In[2]:


dataset = pd.read_csv('project - part D - training data set.csv')
dataset.head()
dataset = dataset.drop('Unnamed: 0', axis = 1)


# In[3]:


testing_dataset = pd.read_csv('project - part D - testing data set.csv')
testing_dataset.head()
testing_dataset = testing_dataset.drop('Unnamed: 0', axis = 1)


# In[4]:


testing_dataset


# In[6]:


dataset.head()


# In[5]:


x = dataset['Father'].values.reshape(-1,1)
y = dataset['Son'].values.reshape(-1,1)


# In[6]:


x_test = testing_dataset['Father'].values.reshape(-1,1)
y_test = testing_dataset['Son'].values.reshape(-1,1)


# In[20]:


plt.hist(x)


# In[21]:


plt.hist(y)


# In[22]:


scatter = plt.scatter(x,y)


# In[35]:


model = Lasso(max_iter=1e5, alpha=1)
poly_feature_generator = preprocessing.PolynomialFeatures(degree=10)
modified_x=poly_feature_generator.fit_transform(x)
model.fit(modified_x,y.ravel())
modified_x_test=poly_feature_generator.fit_transform(x_test)
y_pred = model.predict(modified_x_test)
print('train error', math.sqrt(mean_squared_error(y,model.predict(modified_x))))
print('test error', math.sqrt(mean_squared_error(y_test,y_pred)))

