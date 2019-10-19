
# coding: utf-8

# In[1]:


import gc
import pandas as pd 
import numpy as np 

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import math 
from sklearn import metrics 
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset_train = pd.read_csv('project - part D - training data set.csv')
dataset_train = dataset_train.drop('Unnamed: 0', axis=1)

dataset_test = pd.read_csv('project - part D - testing data set.csv')
dataset_test = dataset_test.drop('Unnamed: 0', axis=1)

#print("\n----- Train Data -----")
#print("Shape         :", dataset_train.shape)
#print("Row Count     :", len(dataset_train))
#print("Column Headers:", dataset_train.columns)
#print("Data Types    :", dataset_train.dtypes)
#print("Index         :", dataset_train.index)
#print("Describe      :", dataset_train.describe())
#print("\n----- Test Data -----")
#print("Shape         :", dataset_test.shape)
#print("Row Count     :", len(dataset_test))
#print("Column Headers:", dataset_test.columns)
#print("Data Types    :", dataset_test.dtypes)
#print("Index         :", dataset_test.index)
#print("Describe      :", dataset_test.describe())


# In[3]:


x_train = dataset_train.iloc[:, :-1]
y_train = dataset_train.iloc[:, -1]
x_test = dataset_test.iloc[:, :-1]
y_test = dataset_test.iloc[:, -1]

x_train = x_train.values.reshape(-1,1)
y_train = y_train.values.reshape(-1,1)
x_test = x_test.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)

x_train = x_train/10000
y_train = y_train/10000
x_test = x_test/10000
y_test = y_test/10000

#scaler = StandardScaler()
#x_train = scaler.fit_transform(x_train)
#x_test = scaler.transform(x_test)

#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)


# In[4]:


train_err = []
test_err = []


# In[5]:


#Due to memory limitation I performed it in manual fashion. Recoreded and polulatated errors in the array. 
gc.collect()
poly = PolynomialFeatures(degree=10)
x_train = poly.fit_transform(x_train)
y_train = poly.fit_transform(y_train)
x_test = poly.fit_transform(x_test)
y_test = poly.fit_transform(y_test)

alpha_vals = np.linspace(1,20,20)
for alpha_v in alpha_vals:
    regr = Lasso(alpha = alpha_v, normalize=True, max_iter=10e5)
    regr.fit(x_train, y_train)
    
    train_err.append(math.sqrt(mean_squared_error(y_train, regr.predict(x_train))))
    test_err.append(math.sqrt(mean_squared_error(y_test, regr.predict(x_test))))


# In[6]:


print (train_err)
print (test_err)


# In[7]:


plt.title('Lasso')
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.plot(np.linspace(0.1,1,20),train_err,'bo-',label='Train')
plt.plot(np.linspace(0.1,1,20),test_err,'ro-',label='Test')
plt.legend()
plt.savefig('Lasso_Reg.png')
plt.show()

