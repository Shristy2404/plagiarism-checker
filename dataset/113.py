import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

df_train = pd.read_csv('train.csv')
X_train=df_train['Father'].values.reshape(-1,1)
y_train=df_train['Son'].values.reshape(-1,1)

df_test = pd.read_csv('test.csv')
X_test=df_test['Father'].values.reshape(-1,1)
y_test=df_test['Son'].values.reshape(-1,1)


train_err=[]
test_err=[]

poly = PolynomialFeatures(degree=10)
modified_x_train=poly.fit_transform(X_train)
modified_x_test=poly.fit_transform(X_test)

reg= Lasso()
reg.fit(modified_x_train,y_train)
lasso_train_rmse=math.sqrt(mean_squared_error(y_train,reg.predict(modified_x_train)))
print("Lasso Train RMSE:",lasso_train_rmse)

reg.fit(modified_x_test,y_test)
lasso_test_rmse=math.sqrt(mean_squared_error(y_test,reg.predict(modified_x_test)))
print("Lasso Test RMSE:",lasso_test_rmse)

reg= Ridge()
reg.fit(modified_x_train,y_train)
ridge_train_rmse=math.sqrt(mean_squared_error(y_train,reg.predict(modified_x_train)))
print("Ridge Train RMSE:",ridge_train_rmse)

reg.fit(modified_x_test,y_test)
ridge_test_rmse=math.sqrt(mean_squared_error(y_test,reg.predict(modified_x_test)))
print("Ridge Test RMSE:",ridge_test_rmse)

reg= LinearRegression()
reg.fit(modified_x_train,y_train)
LinearRegression_train_rmse=math.sqrt(mean_squared_error(y_train,reg.predict(modified_x_train)))
print("LinearRegression Train RMSE:",LinearRegression_train_rmse)


reg.fit(modified_x_test,y_test)
LinearRegression_test_rmse=math.sqrt(mean_squared_error(y_test,reg.predict(modified_x_test)))
print("LinearRegression Test RMSE:",LinearRegression_test_rmse)

if LinearRegression_train_rmse>lasso_train_rmse:
    print('RMSE is greater for LinearRegression compare to Lasso for Train data')
else:
    print('RMSE is less for LinearRegression compare to Lasso for Train data')

if LinearRegression_test_rmse>lasso_test_rmse:
    print('RMSE is greater for LinearRegression compare to Lasso for Test data')
else:
    print('RMSE is less for LinearRegression compare to Lasso for Test data')

if LinearRegression_test_rmse>ridge_train_rmse:
    print('RMSE is greater for LinearRegression compare to Ridge for Train data')
else:
    print('RMSE is less for LinearRegression compare to Ridge for Train data')

if LinearRegression_test_rmse>ridge_test_rmse:
    print('RMSE is greater for LinearRegression compare to Ridge for Test data')
else:
    print('RMSE is less for LinearRegression compare to Ridge for Test data')
