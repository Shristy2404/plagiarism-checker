#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math 


# In[2]:


df= pd.read_csv('E:\Madhav\Machine Learning\Bits Pilani\Assignments\Project D\Project - Part D - C1 (Regression) - PGP in AI ML/project - part D - training data set.csv')
df_test = pd.read_csv('E:\Madhav\Machine Learning\Bits Pilani\Assignments\Project D\Project - Part D - C1 (Regression) - PGP in AI ML/project - part D - testing data set.csv')
x_train = df['Father'].values.reshape(-1,1)
y_train = df['Son'].values.reshape(-1,1)
x_test = df_test['Father'].values.reshape(-1,1)
y_test = df_test['Son'].values.reshape(-1,1)


# In[3]:


poly = PolynomialFeatures(degree=10) 
modified_x_train = poly.fit_transform(x_train)
modified_x_test = poly.fit_transform(x_test)


# In[4]:


reg = Lasso(alpha=0.5,max_iter=70000)
reg.fit(x_train,y_train) 


# In[5]:


print("Lasso Train_RMSE :", math.sqrt(mean_squared_error(y_train,reg.predict(x_train))))
print("Lasso Test_RMSE :",  math.sqrt(mean_squared_error(y_test,reg.predict(x_test))))


# In[18]:


# Finding out best alpha values 

train_err =[]
test_err = []
poly_10 = PolynomialFeatures(degree=10)

modified_x_train = poly_10.fit_transform(x_train)
modified_x_test = poly_10.fit_transform(x_test)

lambda_vals = np.linspace(0,0.1,10)

for m in lambda_vals:

    reg = Lasso(alpha=m)
    reg.fit(modified_x_train,y_train)

    train_err.append(math.sqrt(mean_squared_error(y_train,reg.predict(modified_x_train))))
    test_err.append(math.sqrt(mean_squared_error(y_test,reg.predict(modified_x_test))))



# In[19]:


plt.title('Lasso Regression')
plt.xlabel('Alpha values')
plt.ylabel('RMSE')
plt.plot(np.linspace(0,0.1,10),train_err,'bo-',label='Train')
plt.plot(np.linspace(0,0.1,10),test_err,'ro-',label='Test')
plt.legend()
plt.show()


# In[ ]:




