import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from warnings import filterwarnings
filterwarnings('ignore')

data = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')

array = data.values
X = array[:,0:1]
Y = array[:,1].reshape(-1,1)

poly = PolynomialFeatures(degree=10)
modified_X = poly.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(modified_X, Y, test_size = 0.25, random_state = 53)

model = Lasso()
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

train_rmse = math.sqrt(mean_squared_error(Y_train,model.predict(X_train)))
test_rmse = math.sqrt(mean_squared_error(Y_test,y_pred))
print('Results of Training data split:')
print('Train RMSE:',train_rmse)
print('Test RMSE', test_rmse)


#Validation data
array = test.values
test_X = array[:,0:1]
test_Y = array[:,1].reshape(-1,1)

modified_testX = poly.fit_transform(test_X)
y_pred = model.predict(modified_testX)
test_rmse = math.sqrt(mean_squared_error(test_Y,y_pred))
print('\nResult on validation dataset:')
print('Test RMSE', test_rmse)
print(
'''
Training RMSE of Lasso Regression is almost same as that of Polynomial of degree 10(see polynomial vs rmse graph),
while the testing RMSE of Lasso regression is 1.59 while that of polynomial is 2.03.
Lasso performs slightly better on the test data by penalizing the parameter with appropriate lambda.
''')