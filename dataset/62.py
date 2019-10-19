
# coding: utf-8

# In[20]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
get_ipython().magic('matplotlib inline')


dataset=pd.read_csv("project - part D - training data set.csv")
dataset_test=pd.read_csv("project - part D - testing data set.csv")

x_train=dataset['Father'].values.reshape(-1,1)
y_train=dataset['Son'].values.reshape(-1,1)
x_test=dataset_test['Father'].values.reshape(-1,1)
y_test=dataset_test['Son'].values.reshape(-1,1)

poly_reg=PolynomialFeatures(degree=10)
modified_x_train=poly_reg.fit_transform(x_train)
modified_x_test=poly_reg.fit_transform(x_test)
alpla_vals=np.linspace(0,1,10)
train_err=[]
test_err=[]
for a in alpla_vals:
    poly_reg=Lasso(alpha=a, max_iter=1e7, tol=.001)
    poly_reg.fit(modified_x_train, y_train)
    y_train_predict=poly_reg.predict(modified_x_train)
    y_test_predict=poly_reg.predict(modified_x_test)
    train_err.append(math.sqrt(mean_squared_error(y_train,y_train_predict)))
    test_err.append(math.sqrt(mean_squared_error(y_test,y_test_predict)))
    print("Lasso Train RMSE: ", math.sqrt(mean_squared_error(y_train,y_train_predict)))
    print("Lasso Test RMSE: ", math.sqrt(mean_squared_error(y_test,y_test_predict)))

plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.title('Lasso')
plt.plot(np.linspace(0,1,10),train_err,'bo-',label='Train')
plt.plot(np.linspace(0,1,10),test_err,'ro-', label='Test')
plt.legend()
plt.show()


# In[ ]:

''' Lasso regression gives consistent test RMSE for different values of alpha'''


# In[ ]:




# In[ ]:




# In[ ]:



