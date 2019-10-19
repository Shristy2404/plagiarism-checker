# -*- coding: utf-8 -*-
"""
Created on Thu July 15 14:08:46 2019

@author: Ravikanth S (2018AIML535) Lasso
"""
# the library we will use to create the model
import math
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn import metrics 
from colorama import Fore
 
testds = pd.read_csv('project - part D - testing data set.csv');
trainds = pd.read_csv('project - part D - training data set.csv')

X_train = trainds['Father'].values.reshape(-1,1)
y_train = trainds['Son'].values.reshape(-1,1)
X_test = testds['Father'].values.reshape(-1,1)
y_test = testds['Son'].values.reshape(-1,1)

alpha_vals = np.linspace(0.1,1,10)
tst_RMSE_vals = []
trn_RMSE_vals = []

for alpha_v in alpha_vals:
    poly = preprocessing.PolynomialFeatures(degree=10)
    modified_X = poly.fit_transform(X_train)
    modified_Xtst = poly.fit_transform(X_test)
    
    regr = Lasso(alpha=alpha_v, tol=0.3)
    regr.fit(modified_X,y_train)
    
    trn_mse = metrics.mean_squared_error(y_train,regr.predict(modified_X))
    tst_mse = metrics.mean_squared_error(y_test,regr.predict(modified_Xtst))
    
    
    rmse_trn = math.sqrt(trn_mse)
    rmse_tst = math.sqrt(tst_mse)
    
    tst_RMSE_vals.append(rmse_tst)
    trn_RMSE_vals.append(rmse_trn)
    print(Fore.GREEN+'For Degree 10 Lasso with alpha=%.1f, Train RMSE=%.6f \t Test RMSE=%.6f'%(alpha_v,rmse_trn,rmse_tst)+Fore.RESET)
    
print('=======================================================================================')
print(Fore.GREEN+'For Degree 10 Lasso with default alpha=1.0, Train RMSE=%.6f \t Test RMSE=%.6f'%(trn_RMSE_vals[9],tst_RMSE_vals[9])+Fore.RESET)
print('=======================================================================================')

plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.plot(np.linspace(0.1,1,10),trn_RMSE_vals,'bo--',label='train')
plt.plot(np.linspace(0.1,1,10),tst_RMSE_vals,'ro-.',label='test')
plt.legend()
plt.show()


