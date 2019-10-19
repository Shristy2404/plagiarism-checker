import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
from sklearn.preprocessing import PolynomialFeatures
import math  
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso 

from sklearn import metrics 
dataset_train = pd.read_csv('train.csv')
x_train = dataset_train['Father'].values.reshape(-1,1)
y_train = dataset_train['Son'].values.reshape(-1,1)

dataset_test = pd.read_csv('test.csv')
x_test = dataset_test['Father'].values.reshape(-1,1)
y_test = dataset_test['Son'].values.reshape(-1,1)
 
xDegree =[] 



pFeatures = PolynomialFeatures(degree=10)
x_train_modified = pFeatures.fit_transform(x_train)
x_test_modified = pFeatures.fit_transform(x_test)
ols = LinearRegression( )
ols.fit(x_train_modified, y_train)
linearCoffient =ols.coef_[0].reshape(-1,1)
lasso = Lasso()
lasso.fit(x_train_modified, y_train)
lassoCofficient =lasso.coef_.reshape(-1,1)

y_train_pred = ols.predict(x_train_modified)
y_test_pred = ols.predict(x_test_modified)
rmse_test = math.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
rmse_train = math.sqrt(metrics.mean_squared_error(y_train, y_train_pred))

print("Polynomial REgression RMSE test : ",rmse_test," Train ", rmse_train)


y_train_pred = lasso.predict(x_train_modified)
y_test_pred = lasso.predict(x_test_modified)
rmse_test = math.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
rmse_train = math.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
print("Lasso RMSE test : ",rmse_test," Train ", rmse_train)

for deg in range ( 0 , 11):
    xDegree.append("w "+str(deg))

 

plt.plot(xDegree,linearCoffient,'bo',label='linear Coefficient')
plt.xticks(xDegree) 
plt.plot(xDegree,lassoCofficient,'co' ,label='lasso Coefficient')
plt.legend()
plt.xlabel(' Coefficient Index ')
plt.ylabel('Coefficient Value')
# 
plt.show()    
