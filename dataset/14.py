
# coding: utf-8



# Import all necessary libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math

from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures





#Read test and train data using pandas
trainData = pd.read_csv('training data set.csv')
testData = pd.read_csv('testing data set.csv')





#Read the feature data X
X = trainData.iloc[:, 1:2]
x_test1 = testData.iloc[:,1:2]





#Reshape the data to matrix 
X_train = X.values.reshape(-1,1)
x_test = x_test1.values.reshape(-1,1)





#Read the dependent data y
y = trainData.iloc[:, 2:3]
y_test1 = testData.iloc[:,2:3]





#Reshape the data to matrix 
y_train = y.values.reshape(-1,1)
y_test = y_test1.values.reshape(-1,1)





#Lasso Regression at Degree 10
poly = PolynomialFeatures(degree=10)
modifiedX = poly.fit_transform(X_train)
modifiledx_test = poly.fit_transform(x_test)

reg = Lasso(alpha=0.7)
reg.fit(modifiedX,y_train)
train_err = math.sqrt(mean_squared_error(y_train, reg.predict(modifiedX)))
test_err = math.sqrt(mean_squared_error(y_test, reg.predict(modifiledx_test)))





print ("Lasso Train Error at Degree 10 = ", train_err)
print ("Lasso Test Error at Degree 10 = ", test_err)
print("Poly Regression Train Error at Deg 10 =  1.378712818211837")
print("Poly Regression Test Error at Deg 10 =  1.8179720258261267")




print("Lasso regression's test error (RMSE) is 1.5419812682325975, whereas regular Polynomial regression at Degree 10 has RMSE of 1.8179720258261267")

