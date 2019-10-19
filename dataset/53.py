import numpy as np
import pandas as pd 
import math
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error


#from sklearn.model_selection import train_test_split

dataset_train = pd.read_csv('project - part D - training data set.csv')

x_train= dataset_train['Father']
x_train =x_train.values.reshape(-1,1)

y_train= dataset_train['Son']
y_train =y_train.values.reshape(-1,1)

dataset_test = pd.read_csv('project - part D - testing data set.csv')

x_test= dataset_test['Father']
x_test =x_test.values.reshape(-1,1)

y_test= dataset_test['Son']
y_test =y_test.values.reshape(-1,1)


poly=PolynomialFeatures(degree=10)
modified_x_train=poly.fit_transform(x_train)
modified_x_test=poly.fit_transform(x_test)

print('Lasso\n')
reg = Lasso().fit(modified_x_train,y_train)
print('Train RMSE',  math.sqrt(mean_squared_error(y_train,reg.predict(modified_x_train))))
print('Test RMSE', math.sqrt(mean_squared_error(y_test,reg.predict(modified_x_test))))

#print('Linear Regression \n')
#reg = LinearRegression().fit(modified_x_train,y_train)
#print('Train RMSE',  math.sqrt(mean_squared_error(y_train,reg.predict(modified_x_train))))
#print('Test RMSE', math.sqrt(mean_squared_error(y_test,reg.predict(modified_x_test))))

