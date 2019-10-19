#import libraray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
#import sklearn library
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
#import warning
import warnings
warnings.filterwarnings('ignore')
#read training csv file
train_data = pd.read_csv('project - part D - training data set.csv')
train_data = train_data.drop([train_data.columns[0]], axis=1)
X_train= train_data['Father'].values.reshape(-1,1)
y_train = train_data['Son'].values.reshape(-1,1)
#read testing csv file
test_data = pd.read_csv('project - part D - testing data set.csv')
test_data = test_data.drop([test_data.columns[0]], axis=1)
X_test= test_data['Father'].values.reshape(-1,1)
y_test = test_data['Son'].values.reshape(-1,1)
#Code to plot the graph for Polynomial degree from 1 to 10
train_err =[]
test_err =[]
poly = PolynomialFeatures(degree=10)
modified_X_Train = poly.fit_transform(X_train)
modified_X_Test = poly.fit_transform(X_test)
reg = Lasso(max_iter=10e5)
reg.fit(modified_X_Train,y_train)
print('Lasso Train RMSE with default Lamda with Degree 10:',math.sqrt(mean_squared_error(y_train,reg.predict(modified_X_Train))))
print('Lasso Test RMSE with default Lamda with Degree 10:',math.sqrt(mean_squared_error(y_test,reg.predict(modified_X_Test))))
train_err =[]
test_err =[]
alpha_vals = np.linspace(0,1,10)
for alpha_v in alpha_vals:
    reg = Lasso(alpha=alpha_v, max_iter=10e5)
    reg.fit(modified_X_Train,y_train)
    train_err.append(math.sqrt(mean_squared_error(y_train,reg.predict(modified_X_Train))))
    test_err.append(math.sqrt(mean_squared_error(y_test,reg.predict(modified_X_Test))))
plt.title("Lasso")
plt.xlabel('Alpha Value')
plt.ylabel('RMSE')
plt.plot(np.linspace(0,1,10),train_err,'bo-',label='Train')
plt.plot(np.linspace(0,1,10),test_err,'ro-',label='Test')
plt.legend()
plt.show()
