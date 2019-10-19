import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

# %matplotlib inline  
import math 

# the library we will use to create the model 
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn import metrics 


train_data=pd.read_csv("train.csv")
x_train=train_data['Father'].values.reshape(-1,1)
y_train=train_data['Son'].values.reshape(-1,1)

test_data=pd.read_csv("test.csv")
x_test=test_data['Father'].values.reshape(-1,1)
y_test=test_data['Son'].values.reshape(-1,1)

polyreg = PolynomialFeatures(degree=10)
x_modified_train = polyreg.fit_transform(x_train)
x_modified_test = polyreg.fit_transform(x_test)
model= linear_model.Lasso(alpha=0.5)
model.fit(x_modified_train, y_train)
y_predicted_test=model.predict(x_modified_test)
y_predicted_train=model.predict(x_modified_train)
print('RMSE Training:',sqrt(mean_squared_error(y_train, y_predicted_train)))
print('RMSE Testing:',sqrt(mean_squared_error(y_test, y_predicted_test)))

train_err =[]
test_err=[]
alpha_vals=np.linspace(0,1,10)
for alpha_v in alpha_vals:
    polyreg=linear_model.Lasso(alpha=alpha_v)
    polyreg.fit(x_train,y_train)
    train_err.append(sqrt(mean_squared_error(y_train, polyreg.predict(x_train))))
    test_err.append(sqrt(mean_squared_error(y_test, polyreg.predict(x_test))))
plt.title('Lasso Regression')
plt.xlabel('Alpha value')
plt.ylabel('RMSE')
plt.plot(np.linspace(0,1,10),train_err,'bo-',label='Training')
plt.plot(np.linspace(0,1,10),test_err,'ro-',label='Testing')
plt.legend()
plt.show()