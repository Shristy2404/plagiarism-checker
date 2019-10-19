import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import math
import matplotlib.pyplot as plt
train_data=pd.read_csv('C:\\Users\\DELL\\Desktop\\Assignments\\train.csv')
test_data=pd.read_csv('C:\\Users\\DELL\\Desktop\\Assignments\\test.csv')
#print(train_data)
x_train=train_data['Father'].values.reshape(-1,1)
y_train=train_data['Son'].values.reshape(-1,1)
x_test=test_data['Father'].values.reshape(-1,1)
y_test=test_data['Father'].values.reshape(-1,1)
train_error=[]
test_error=[]
for i in np.linspace(0,1,10):
    poly=PolynomialFeatures(degree=10)
    modified_x_train=poly.fit_transform(x_train)
    modified_x_test=poly.fit_transform(x_test)
    reg=Lasso(alpha=i)
    reg.fit(modified_x_train,y_train)
    y_train_pred=reg.predict(modified_x_train)
    y_test_pred=reg.predict(modified_x_test)
    train_mse=mean_squared_error(y_train,y_train_pred)
    train_rmse=math.sqrt(train_mse)
    test_mse=mean_squared_error(y_test,y_test_pred)
    test_rmse=math.sqrt(test_mse)
    train_error.append(train_rmse)
    test_error.append(test_rmse)
#print(train_error,test_error)
plt.plot(np.linspace(0,1,10),train_error,'bo-',label='train')
plt.plot(np.linspace(0,1,10),test_error,'ro-',label='test')
plt.legend()
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.show()
print('Best alpha value is 0.9')
print(' improvements brought down by Lasso regression in RMSE as compared to the typical regression model built using the polynomial of degree 10 is:\n Reduces overfitting problem: In Normal linear regression at degree 10 the test error is huge  compared to the training error however that didnt happen in Ridge regression')

    