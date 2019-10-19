#!/usr/bin/env python
# coding: utf-8

# # Program to implement Lasso Regression

# In[173]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# # # Step 1
# Read Training and Testing Data from specified files

# In[134]:


training_data_set = pd.read_csv('project - part D - training data set.csv')
testing_data_set = pd.read_csv('project - part D - testing data set.csv')
x_train = training_data_set['Father'].values.reshape(-1,1)
y_train = training_data_set['Son'].values.reshape(-1,1)
x_test = testing_data_set['Father'].values.reshape(-1,1)
y_test = testing_data_set['Son'].values.reshape(-1,1)


# Below function converts array with single feature x to a transformed array with feature set as x^2, x^3 upto required degree.

# In[135]:


def obtain_polynomial_x(x, degree_required):
    poly = PolynomialFeatures(degree=degree_required, include_bias=False)
    modified_x = poly.fit_transform(x)
    return modified_x


# Below function implements L2 regularization i.e. Lasso Regression on the data points given.

# In[136]:


def lasso_reg_runner(max_degree, x_train, y_train, x_test, y_test, train_err, test_err ):
    result=[]
    for i in range(1, max_degree+1):
        modified_x_train = obtain_polynomial_x(x_train, i)
        modified_x_test = obtain_polynomial_x(x_test, i)
        reg = Lasso().fit(modified_x_train, y_train)
        y_train_predict = reg.predict(modified_x_train)
        train_data_mse = mean_squared_error(y_train_predict, y_train)
        train_data_rmse = math.sqrt(train_data_mse)
        y_test_predict = reg.predict(modified_x_test)
        test_data_mse = mean_squared_error(y_test_predict, y_test)
        test_data_rmse = math.sqrt(test_data_mse)
        train_err.append(train_data_rmse)
        test_err.append(test_data_rmse)
        result.append({'degree': i,
                       'coefficients': reg.coef_,
                       'intercept': reg.intercept_,
                       'trainRMSE': train_data_rmse,
                       'testRMSE': test_data_rmse})
    return result;


# Below function implements Normal Regression on the given data points

# In[137]:


def polynomial_reg_runner(max_degree, x_train, y_train, x_test, y_test, train_err, test_err):
    result = []
    for i in range(1, max_degree+1):
        modified_x_train = obtain_polynomial_x(x_train, i)
        modified_x_test = obtain_polynomial_x(x_test, i)
        reg = LinearRegression().fit(modified_x_train, y_train)
        y_train_predict = reg.predict(modified_x_train)
        train_data_mse = mean_squared_error(y_train_predict, y_train)
        train_data_rmse = math.sqrt(train_data_mse)
        y_test_predict = reg.predict(modified_x_test)
        test_data_mse = mean_squared_error(y_test_predict, y_test)
        test_data_rmse = math.sqrt(test_data_mse)
        train_err.append(train_data_rmse)
        test_err.append(test_data_rmse)
        result.append({'degree': i,
                       'coefficients': reg.coef_,
                       'intercept': reg.intercept_,
                       'trainRMSE': train_data_rmse,
                       'testRMSE': test_data_rmse})
    return result;


# # Step 3
# Run Lasso and Normal Regression on the data points and plot test and train error to visualize the effects of Lasso Regularization on the test and train errors. By the plot generated it can be seen that the tesing error after L2 regularization for the regression model is lesser than what we have in Regular regression model.

# In[138]:


train_err_lasso=[]
test_err_lasso=[]
train_err_regular=[]
test_err_regular=[]
lasso_result = lasso_reg_runner(10, x_train, y_train, x_test, y_test, train_err_lasso, test_err_lasso)
regular_result = polynomial_reg_runner(10, x_train, y_train, x_test, y_test, train_err_regular, test_err_regular)
plt.figure(num=None, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
plt.xlabel('Polynomial Degree')
plt.ylabel('RMSE')
plt.plot(range(1,11), train_err_lasso, 'bo-', label='Train Error With Lasso')
plt.plot(range(1,11), test_err_lasso, 'ro-', label='Test Error With Lasso')
plt.plot(range(1,11), test_err_regular, 'go-', label='Test Error With Regular')
plt.plot(range(1,11), train_err_regular, 'yo-', label='Train Error With Regular')
plt.title('RMSE vs Degree of Model Selected')
plt.legend()
plt.grid(True)
plt.savefig('2018AIML646_lasso_part_d.png')


# # Step 4
# To visualize further the effects of Lasso Regression plot the obtained coefficients and intercepts of Lasso Regression as well as Normal Regression model. From the plots it can be inferred that in Normal Regression model the coefficients and intercept are given enough freedom to grow as large as possible so that the curve may try to pass through every datapoint in training data set, however this results in a very high testing error.
# 
# However, with L2 regularization or Lasso regularization these coefficients now don't have that much freedom to grow large enough and increase the testing error. Thus, combatting overfitting problems of regression models.

# In[171]:


lasso_coefficient_matrix = []
lasso_intercept_matrix = []
regular_coefficient_matrix = []
regular_intercept_matrix = []
#figures = []
for j in range(10):
    lasso_coefficient_matrix.append(np.concatenate((lasso_result[j]['coefficients'], np.zeros(9-j)) ))
    regular_coefficient_matrix.append( np.concatenate((regular_result[j]['coefficients'][0], np.zeros(9-j)) ))
    lasso_intercept_matrix.append(lasso_result[j]['intercept'])
    regular_intercept_matrix.append(regular_result[j]['intercept'])
lasso_coefficient_matrix = np.asarray(lasso_coefficient_matrix)
regular_coefficient_matrix = np.asarray(regular_coefficient_matrix)
fig,ax = plt.subplots(5, 2, sharex='all', figsize=(20, 20))
plot_index = [0,0]
for j in range(10):
    plt_ax = ax[plot_index[0],plot_index[1]]
    plt_ax.set_xlabel('Polynomial Degree')
    plt_ax.set_ylabel('Coeficient w'+str(j+1))
    plt_ax.plot(range(1,11),lasso_coefficient_matrix[:,j], 'bo-', label='Lasso')
    plt_ax.plot(range(1,11),regular_coefficient_matrix[:,j], 'ro-', label='regular')
    plt_ax.legend()
    plt_ax.grid(True)
    if j%2 == 0:
        plot_index[1]+=1
    else:
        plot_index[0]+=1
        plot_index[1]-=1
plt.savefig('2018AIML646_lasso_coefficients_part_d.png')


# In[172]:


plt.xlabel('Polynomial Degree')
plt.ylabel('Intercept w0')
plt.plot(range(1,11),lasso_intercept_matrix, 'bo-', label='Lasso')
plt.plot(range(1,11),regular_intercept_matrix, 'ro-', label='regular')
plt.legend()
plt.grid(True)
plt.savefig('2018AIML646_lasso_intercepts_part_d.png')


# In[174]:


print('The training RMSE for Degree 10 after Lasso Regularization is : '+ str(lasso_result[9]['trainRMSE']))
print('The testing RMSE for Degree 10 after Lasso Regularization is : '+ str(lasso_result[9]['testRMSE']))


# In[ ]:




