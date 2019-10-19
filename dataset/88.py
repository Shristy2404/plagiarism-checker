import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

training=pd.read_csv('train_data.csv')
testing =pd.read_csv('test_data.csv')

x_train =training['Father'].values.reshape(-1,1)
y_train =training['Son'].values.reshape(-1,1)

x_test =testing['Father'].values.reshape(-1,1)
y_test =testing['Son'].values.reshape(-1,1)

train_err = []
test_err  = []

for i in range (1,11):
    poly = PolynomialFeatures(degree =i)
    modified_X_train = poly.fit_transform(x_train)
    modified_X_test = poly.fit_transform(x_test)
    
    if i==1:
        print('Training Points: ',x_train.shape[0])
        print('Testing Points: ',x_test.shape[0])
        
    
    alpha_vals = np.linspace(0,1,10)
    for alpha_v in alpha_vals:
        reg = Lasso(alpha=alpha_v) 
        reg.fit(modified_X_train,y_train)
    
    train_err.append(math.sqrt(mean_squared_error(y_train,reg.predict(modified_X_train))))
    test_err.append(math.sqrt(mean_squared_error(y_test,reg.predict(modified_X_test))))
    
    train_RMSE =math.sqrt(mean_squared_error(y_train,reg.predict(modified_X_train)))
    test_RMSE  =math.sqrt(mean_squared_error(y_test,reg.predict(modified_X_test)))
    
print('Lasso Train RMSE : ' ,train_RMSE)
print('Lasso Test RMSE : ' ,test_RMSE)
      
plt.xlabel('Polynomial Degree')
plt.ylabel('RMSE')
plt.plot(np.linspace(1,11,10),train_err,'bo-',label='Train')
plt.plot(np.linspace(1,11,10),test_err,'ro-',label='Test')
plt.legend()
plt.show()