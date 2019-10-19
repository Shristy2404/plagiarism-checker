#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')


# In[29]:


test_data = pd.read_csv('testing_data_set.csv')
train_data = pd.read_csv('training_data_set.csv')


# In[30]:


x_train = train_data['Father'].values.reshape(-1,1)
y_train = train_data['Son'].values.reshape(-1,1)
x_test = test_data['Father'].values.reshape(-1,1)
y_test = test_data['Son'].values.reshape(-1,1)


# In[31]:


poly = PolynomialFeatures(degree = 10)
X_modified_train = poly.fit_transform(x_train)
X_modified_test = poly.fit_transform(x_test)
model1 = linear_model.Lasso(alpha = 0.5)
model1.fit(X_modified_train,y_train)
y_predicted_test = model1.predict(X_modified_test)
y_predicted_train = model1.predict(X_modified_train)
a = sqrt(mean_squared_error(y_train,y_predicted_train))
b = sqrt(mean_squared_error(y_test,y_predicted_test))
print('Lasso Train RMSE:',a)
print('Lasso Test RMSE:',b)

    


# In[26]:


train_err = []
test_err = []
alpha_vals = np.linspace(0,1,9)
for alpha_v in alpha_vals:
    polyreg = linear_model.Lasso(alpha = alpha_v)
    polyreg.fit(x_train,y_train)
    train_err.append(sqrt(mean_squared_error(y_train,polyreg.predict(x_train))))
    test_err.append(sqrt(mean_squared_error(y_test,polyreg.predict(x_test))))
plt.title('Lasso')
plt.xlabel('Alpha value')
plt.ylabel('RMSE')
plt.plot(np.linspace(0,1,9),train_err,'bo-',label = 'Train')
plt.plot(np.linspace(0,1,9),test_err,'ro-',label = 'Test')
plt.legend()
plt.show()


# In[32]:


print('Ridge regression for polynomial of degree 10 did not bring any improvements to the RMSE of the training data but shown improvements in the RMSE of testing data with default alpha as compared to polynomial regression of degree 10')


# In[ ]:




