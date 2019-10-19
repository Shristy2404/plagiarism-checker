import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

import math

# %matplotlib inline

DATAPATH = 'project - part D - training data set.csv'

datasetTrain = pd.read_csv(DATAPATH, delimiter = ',')

X = datasetTrain['Father'].values/10000
X = X.reshape(-1,1)
y = datasetTrain['Son'].values.reshape(-1,1)

datasetTest = pd.read_csv('project - part D - testing data set.csv', delimiter = ',')

X1 = datasetTest['Father'].values/10000
X1 = X1.reshape(-1,1)
y1 = datasetTest['Son'].values.reshape(-1,1)

poly = PolynomialFeatures(degree = 10) 
modified_X = poly.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(modified_X,y,test_size=0.27, random_state = 1)

reg = Lasso(alpha=0.5)#regulating parameter
reg.fit(X_train,y_train)

print('Lasso Train RMSE: ', math.sqrt(mean_squared_error(y_train,reg.predict(X_train))))
print('Lasso Test RMSE: ', math.sqrt(mean_squared_error(y_test,reg.predict(X_test))))

reg = Lasso(alpha=0.5)#regulating parameter
reg.fit(X,y)

print('Lasso Train RMSE1: ', math.sqrt(mean_squared_error(y,reg.predict(X))))
print('Lasso Test RMSE1: ', math.sqrt(mean_squared_error(y1,reg.predict(X1))))

#Lasso Regularization
train_err = []
test_err = []
poly_10 = PolynomialFeatures(degree = 10) 
modified_X = poly_10.fit_transform(X)
#print('modified_X: ', modified_X)

X_train,X_test,y_train,y_test = train_test_split(modified_X,y,test_size=0.27, random_state = 1)

alpha_vals = np.linspace(0,1,10)
#print('alpha_vals: ', alpha_vals)
for alpha_v in alpha_vals:
    reg = Lasso(alpha=alpha_v)
    reg.fit(X_train,y_train)
    
    train_err.append(math.sqrt(mean_squared_error(y_train,reg.predict(X_train))))
    test_err.append(math.sqrt(mean_squared_error(y_test,reg.predict(X_test))))
    
plt.title('Lasso Regularization')
plt.xlabel('Alpha Value')
plt.ylabel('RMSE')
plt.plot(np.linspace(0,1,10),train_err,'bo-',label='Train')
plt.plot(np.linspace(0,1,10),test_err,'ro-',label='Test')
plt.legend()
plt.show()