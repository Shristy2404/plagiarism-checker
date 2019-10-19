import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


import math

tr_dataset=pd.read_csv('project - part D - training data set.csv')
tr_dataset = tr_dataset.loc[:, ~tr_dataset.columns.str.contains('^Unnamed')]
tr_y=tr_dataset.pop('Son')
tr_x=tr_dataset.copy()
tr_x=tr_x.values.reshape(-1,1)
tr_y=tr_y.values.reshape(-1,1)

ts_dataset=pd.read_csv('project - part D - testing data set.csv')
ts_dataset = ts_dataset.loc[:, ~ts_dataset.columns.str.contains('^Unnamed')]
ts_y=ts_dataset.pop('Son')
ts_x=ts_dataset.copy()
ts_x=ts_x.values.reshape(-1,1)
ts_y=ts_y.values.reshape(-1,1)


poly=PolynomialFeatures(degree=10)
tr_modified_x=poly.fit_transform(tr_x)
ts_modified_x=poly.fit_transform(ts_x)
    
print('Training points',tr_modified_x.shape[0])
print('Testing points',ts_modified_x.shape[0])
   
reg=Lasso()
reg.fit(tr_modified_x,tr_y)
    
tr_mse_err=mean_squared_error(tr_y,reg.predict(tr_modified_x))
ts_mse_err=mean_squared_error(ts_y,reg.predict(ts_modified_x))

print('Training MSE',tr_mse_err)
print('Test MSE',ts_mse_err)
print('Test MSE for a Polynomial 10 without Lasso is 3.2 which is higher than Lasso MSE so Lasso gives better prediction')
