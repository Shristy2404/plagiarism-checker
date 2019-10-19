import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import math



best_polynomial={"Alpha":0,"RMSE":0}

dataset=pd.read_csv('project - part D - training data set.csv')
X_train = dataset['Father'].values.reshape(-1,1)
y_train = dataset['Son'].values.reshape(-1,1)

dataset=pd.read_csv('project - part D - testing data set.csv')
X_test = dataset['Father'].values.reshape(-1,1)
y_test = dataset['Son'].values.reshape(-1,1)

poly=PolynomialFeatures(degree=10)
X_train_modified=poly.fit_transform(X_train)
X_test_modified=poly.fit_transform(X_test)

ols=LinearRegression()
ols.fit(X_train_modified,y_train)
Train_RMSE=math.sqrt(mean_squared_error(y_train,ols.predict(X_train_modified)))
Test_RMSE=math.sqrt(mean_squared_error(y_test,ols.predict(X_test_modified)))   


print("OLS TrainRMSE= ",Train_RMSE,"TestRMSE",Test_RMSE)


lassoReg=Lasso()
lassoReg.fit(X_train_modified,y_train)

Train_RMSE=math.sqrt(mean_squared_error(y_train,lassoReg.predict(X_train_modified)))
Test_RMSE=math.sqrt(mean_squared_error(y_test,lassoReg.predict(X_test_modified)))   


print("Lasso TrainRMSE= ",Train_RMSE,"TestRMSE",Test_RMSE)

parameters_lasso=lassoReg.coef_.reshape(-1,1)
parameters_ols=ols.coef_.reshape(-1,1)




plt.title('Lasso Regression')
plt.xlabel('Alpha Value')
plt.ylabel('RMSE')
plt.plot(Train_RMSE,'bo-',label='Train')
plt.plot(Test_RMSE,'ro-',label='Test')
plt.legend()
plt.show()

plt.plot(parameters_lasso,'bo-',label='Lasso Coefficients')
plt.plot(parameters_ols,'ro-',label='Ols Coefficients')
plt.legend()
plt.show()