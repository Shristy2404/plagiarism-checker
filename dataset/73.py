import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import SGDRegressor 
from sklearn import preprocessing
from sklearn.preprocessing import  StandardScaler
from sklearn.preprocessing import  PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# %matplotlib inline  

import math
import warnings
warnings.filterwarnings('ignore')

df1 = pd.read_csv('project - part D - training data set.csv')
x_train = df1['Father']
x_train = np.array(x_train).reshape(-1,1)

y_train = df1['Son']
y_train = np.array(y_train).reshape(-1,1)

df2 = pd.read_csv('project - part D - testing data set.csv')
x_test = df2['Father']
x_test = np.array(x_test).reshape(-1,1)

y_test = df2['Son']
y_test = np.array(y_test).reshape(-1,1)


from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

poly_10 = PolynomialFeatures(degree=10)
modified_x_train = poly_10.fit_transform(x_train)
modified_x_test = poly_10.fit_transform(x_test)


reg_lr = LinearRegression()
reg_lr.fit(modified_x_train,y_train)
print("Linear Regression - Polynomial degree 10 Training RMSE:",math.sqrt(mean_squared_error(y_train,reg_lr.predict(modified_x_train))))
print("Linear Regression - Polynomial degree 10 Testing RMSE:",math.sqrt(mean_squared_error(y_test,reg_lr.predict(modified_x_test))))



reg10 = Lasso()
reg10.fit(modified_x_train,y_train)
print("Lasso Training RMSE:",math.sqrt(mean_squared_error(y_train,reg10.predict(modified_x_train))))
print("Lasso Testing RMSE:",math.sqrt(mean_squared_error(y_test,reg10.predict(modified_x_test))))


train_err = []
test_err = []

alpha_vals = np.linspace(0,1,9)
for alpha_v in alpha_vals:
    reg = Lasso(alpha=alpha_v)
    reg.fit(modified_x_train,y_train)
    train_err.append(math.sqrt(mean_squared_error(y_train,reg.predict(modified_x_train))))
    test_err.append(math.sqrt(mean_squared_error(y_test,reg.predict(modified_x_test))))
    print("Alpha:",alpha_v,"Lasso Training RMSE:",math.sqrt(mean_squared_error(y_train,reg.predict(modified_x_train))))
    print("Alpha:",alpha_v,"Lasso Testing RMSE:",math.sqrt(mean_squared_error(y_test,reg.predict(modified_x_test))))
    
plt.ylabel('RMSE Lasso')
plt.xlabel('Alpha Value')
plt.plot(np.linspace(0,1,9),train_err,'bo-',label='Train')
plt.plot(np.linspace(0,1,9),test_err,'ro-',label='Test')
plt.legend()
plt.show()
