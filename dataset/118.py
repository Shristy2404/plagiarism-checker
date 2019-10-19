# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 22:27:58 2019

@author: Sameer
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# Remember what this line did?
#get_ipython().run_line_magic('matplotlib', 'inline')
import math 

# the library we will use to create the model 
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures 
import warnings

warnings.filterwarnings('ignore')

def model_performance_params(result,X_train,y_train):
    
    #y_pred = np.polyval(result, X_train)
    y_pred = result.predict(X_train)
    mse = mean_squared_error(y_train,y_pred)
    rmse = math.sqrt(mse)
    rse = mse*(y_train.size)
    rse = rse/(y_train.size-2)
    rse = math.sqrt(rse)
    r_square = metrics.r2_score(y_train,y_pred)
    return rmse,rse,r_square

def draw_plot(result,X,y):
    x_viz = np.linspace(X.min(),X.max())
    plt.plot(x_viz, result.predict(x_viz[:, np.newaxis]), label="Model")
    plt.legend(['Train','Test'])
    plt.show()


path = 'C:\\Users\\Sameer\Documents\\Personal\Bits Pillani\\Course\\DataSets\\'
file_name = 'project - part D - training data set.csv'
dataset = pd.read_csv(file_name)

#X_train,X_test,y_train,y_test = train_test_split(dataset['Father'],dataset['Son'],test_size=0.3,shuffle=False)

X_train = dataset['Father']
y_train = dataset['Son']

#X_train = preprocessing.scale(X_train)
#y_train = preprocessing.scale(y_train)

file_name = 'project - part D - testing data set.csv'
dataset = pd.read_csv(file_name)

X_test = dataset['Father']
y_test = dataset['Son']

#X_test = preprocessing.scale(X_test)
#y_test = preprocessing.scale(y_test)


X_train = np.array(X_train,dtype=float).reshape(-1,1)

X_test = np.array(X_test,dtype=float).reshape(-1,1)

y_train = np.array(y_train,dtype=float).reshape(-1,1)

y_test = np.array(y_test,dtype=float).reshape(-1,1)

polynomial_degree = 10
rmse_train = []
rmse_test = []
popt_train = []

polynomial_features = PolynomialFeatures(degree=10)
X_train_mod = polynomial_features.fit_transform(X_train)
X_test_mod = polynomial_features.fit_transform(X_test)


model = Lasso(alpha=0.5)
result = model.fit(X_train_mod,y_train)
rmse,rse,r_square = model_performance_params(result,X_train_mod,y_train) 
print (result.coef_)
#print (rmse,rse,r_square)
print ('Lasso Model')
print ('Training RMSE :' ,rmse)

rmse,rse,r_square = model_performance_params(result,X_test_mod,y_test) 
print ('Test RMSE :' ,rmse)


'''
model = Ridge(alpha=0.5)
result = model.fit(X_train,y_train)
rmse,rse,r_square = model_performance_params(result,X_train,y_train) 
print (result.coef_)
print (rmse,rse,r_square)
'''
