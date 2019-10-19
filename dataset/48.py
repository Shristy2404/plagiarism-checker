#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

#Read the input file
train_dataset=pd.read_csv(r"C:\Users\madhukiran_pala\Desktop\Project_D_input_train.csv", delimiter = ',')
x_train= train_dataset['Father'].values.reshape(-1,1)
y_train = train_dataset['Son'].values.reshape(-1,1)
test_dataset=pd.read_csv(r"C:\Users\madhukiran_pala\Desktop\Project_D_input_test.csv", delimiter = ',')
x_test= test_dataset['Father'].values.reshape(-1,1)
y_test = test_dataset['Son'].values.reshape(-1,1)
#######################
poly_features = PolynomialFeatures(degree=10)
modified_X_train = poly_features.fit_transform(x_train)
modified_X_test = poly_features.fit_transform(x_test)
rr = Lasso(alpha=1) # 
rr.fit(modified_X_train, y_train)
##########
y_pred_train = rr.predict(modified_X_train)
y_pred_test = rr.predict(modified_X_test)

test_mse = mean_squared_error(y_test, y_pred_test)
train_mse = mean_squared_error(y_train, y_pred_train)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
print("train_rmse" ,train_rmse )
print("test_rmse" ,test_rmse )


# In[ ]:




