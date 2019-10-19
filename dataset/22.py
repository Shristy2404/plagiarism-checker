import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

 
import math 

# the library we will use to create the model 
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

from sklearn.model_selection import train_test_split
from sklearn import metrics 
traindata=pd.read_csv("train.csv")
testdata=pd.read_csv("test.csv")
x_train=traindata['Father'].values.reshape(-1,1)
y_train=traindata['Son'].values.reshape(-1,1)
x_test=testdata['Father'].values.reshape(-1,1)
y_test=testdata['Son'].values.reshape(-1,1)
from sklearn.metrics import mean_squared_error
from math import sqrt
polyreg = PolynomialFeatures(degree=10)
x_modified_train = polyreg.fit_transform(x_train)
x_modified_test = polyreg.fit_transform(x_test)
model= linear_model.Lasso(alpha=0.05,max_iter=10000,tol=.05)
model.fit(x_modified_train, y_train)
y_predicted_test=model.predict(x_modified_test)
y_predicted_train=model.predict(x_modified_train)
print('RMSE Train:',sqrt(mean_squared_error(y_train, y_predicted_train)))
print('RMSE Test:',sqrt(mean_squared_error(y_test, y_predicted_test)))

train_err =[]

test_err=[]

alpha_vals=np.linspace(0,1,10)

for alpha_v in alpha_vals:

    reg=linear_model.Lasso(alpha=alpha_v)
    
    reg.fit(x_train,y_train)

    train_err.append(sqrt(mean_squared_error(y_train, reg.predict(x_train))))

    test_err.append(sqrt(mean_squared_error(y_test, reg.predict(x_test))))

plt.title('Lasso')

plt.xlabel('Alpha value')

plt.ylabel('RMSE')

plt.plot(np.linspace(0,1,10),train_err,'bo-',label='Train')

plt.plot(np.linspace(0,1,10),test_err,'ro-',label='Test')

plt.legend()

plt.show()

