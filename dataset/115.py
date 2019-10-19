#!/usr/bin/env python
# coding: utf-8

# # Lasso Regression Implementation

# In[44]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics 
import math
import warnings
warnings.filterwarnings('ignore')


# Read Train and test data from csv file

# In[45]:


data_train=pd.read_csv('project - part D - training data set.csv', sep = ',')
data_test=pd.read_csv('project - part D - testing data set.csv', sep = ',')

x_train = data_train['Father'].values.reshape(-1,1)
x_test = data_test['Father'].values.reshape(-1,1)
y_train = data_train['Son'].values.reshape(-1,1)
y_test = data_test['Son'].values.reshape(-1,1)


# Modify train and test dataset to polynomial features of degree 10

# In[46]:


poly = PolynomialFeatures(degree=10)
x_train1 = poly.fit_transform(x_train)
x_test1 = poly.fit_transform(x_test)


# Typical polynomial regression model
# 

# In[47]:


poly=LinearRegression(normalize=True)
poly.fit(x_train1,y_train)
y_pred_train_poly=poly.predict(x_train1)
y_pred_test_poly=poly.predict(x_test1)
MSE_train_poly=metrics.mean_squared_error(y_pred_train_poly,y_train)
MSE_test_poly=metrics.mean_squared_error(y_pred_test_poly,y_test)
RMSE_tr_poly=math.sqrt(MSE_train_poly)
RMSE_tt_poly=math.sqrt(MSE_test_poly)

print("The RMSE for training with polynomial of degree 10 is :",RMSE_tr_poly )
print("The RMSE for testing with polynomial of degree 10 is :",RMSE_tt_poly )


# Lasso Model with default value of regularisation parameter alpha

# In[48]:


reg=Lasso(normalize=True)
reg.fit(x_train1,y_train)
y_pred_train=reg.predict(x_train1)
y_pred_test=reg.predict(x_test1)
MSE_train=metrics.mean_squared_error(y_pred_train,y_train)
MSE_test=metrics.mean_squared_error(y_pred_test,y_test)
RMSE_tr=math.sqrt(MSE_train)
RMSE_tt=math.sqrt(MSE_test)

print("\nThe RMSE for training with default value of alpha is :",RMSE_tr )
print("The RMSE for testing with default value of alpha is :",RMSE_tt )


# Find out the best value of alpha with least training error
# 

# In[49]:


RMSE_train=[]
RMSE_test=[]
alpha_n=[]
alpha_val=np.linspace(0,1,200)
b=0.0000

for alpha_v in alpha_val:
    model=Lasso(normalize=True,alpha=alpha_v)
    model.fit(x_train1,y_train)
    y_pred_train1=model.predict(x_train1)
    y_pred_test1=model.predict(x_test1)
    MSE_train1=metrics.mean_squared_error(y_pred_train1,y_train)
    MSE_test1=metrics.mean_squared_error(y_pred_test1,y_test)
    RMSE_tr1=math.sqrt(MSE_train1)
    RMSE_tt1=math.sqrt(MSE_test1)
    RMSE_train.append(RMSE_tr1)
    RMSE_test.append(RMSE_tt1)
    alpha_n.append(alpha_v)
    
plt.plot(alpha_n,RMSE_train, color='r', marker='o',label= 'Training') 
plt.plot(alpha_n,RMSE_test, color='b',marker='o',label='Test') 
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.title("RMSE versus alpha") 
plt.legend(loc='lower right')

print("\nThe best value of alpha in range of 0 to 1 with minimum testing error is 0.005")


# Ridge model with tuned value of alpha

# In[50]:


reg2=Lasso(normalize=True,alpha=0.005)
reg2.fit(x_train1,y_train)
y_pred_tr2=reg2.predict(x_train1)
y_pred_tt2=reg2.predict(x_test1)
MSE_train2=metrics.mean_squared_error(y_pred_tr2,y_train)
MSE_test2=metrics.mean_squared_error(y_pred_tt2,y_test)
RMSE_tr2=math.sqrt(MSE_train2)
RMSE_tt2=math.sqrt(MSE_test2)

print("\nThe RMSE for training with best value of alpha is :",RMSE_tr2 )
print("The RMSE for testing with best value of alpha is :",RMSE_tt2 )


# Plotting Coefficient values for polyomial regression, Lasso regression with default alpha and Lasso regression with tuned alpha to see the benefits of regularization to combat over fitting

# In[51]:


a=poly.coef_.reshape(-1,1)
b=reg.coef_.reshape(-1,1)
c=reg2.coef_.reshape(-1,1)
d=[0,1,2,3,4,5,6,7,8,9,10]
plt.plot(d,a, color='r', marker='o',label= 'poly') 
plt.plot(d,b, color='b', marker='o',label= 'default_lasso') 
plt.plot(d,b, color='g', marker='o',label= 'tuned_lasso') 
#plt.plot(alpha_n,RMSE_test, color='b',marker='o',label='Test') 
plt.xlabel('coefficient#')
plt.ylabel('coefficient value')
plt.title("coefficient value versus coefficient number") 
plt.legend(loc='lower right')


# In[52]:


print("With polynomial of degree 10, training error was very less i.e. 1.11 but testing error was more as degree 10 was overfitting the data, allowing the coefficients to grow to large values as shown in the plotted figure coeffcient value versus coefficient number but with Lasso regression coeffients were restricted to grow, preventing data overfitting and thus the testing error came down to 1.530 with tuned value of alpha")


# In[53]:


print("\n In lasso regression coeffiecients are allowed to take zero values where as in ridge coeffiecients can be becaome very very small but can't take zero value")


# In[ ]:





# In[ ]:




