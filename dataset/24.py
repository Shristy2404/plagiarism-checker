
## LASSO REGRESSION  ##


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#%matplotlib inline  
import math 
from sklearn.metrics import mean_squared_error
from math import sqrt
# the library we will use to create the model 
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics 



def lasso_reg():
    traindata=pd.read_csv("project - part D - training data set.csv")
    testdata=pd.read_csv("project - part D - testing data set.csv")
    x_train=traindata['Father'].values.reshape(-1,1)
    y_train=traindata['Son'].values.reshape(-1,1)
    x_test=testdata['Father'].values.reshape(-1,1)
    y_test=testdata['Son'].values.reshape(-1,1)
    
    polyreg = PolynomialFeatures(degree=10)
    x_modified_train = polyreg.fit_transform(x_train)
    x_modified_test = polyreg.fit_transform(x_test)
    model= linear_model.Lasso(alpha=0.5)
    model.fit(x_modified_train, y_train)
    y_predicted_test=model.predict(x_modified_test)
    y_predicted_train=model.predict(x_modified_train)
    print('Train RMSE for Lasso with Polynoial Degree 10:',sqrt(mean_squared_error(y_train, y_predicted_train)))
    print('Test RMSE for Lasso with Polynoial Degree 10::',sqrt(mean_squared_error(y_test, y_predicted_test)))
    train_err =[]
    test_err=[]
    alpha_vals=np.linspace(0,1,20)
    for alpha_v in alpha_vals:
        #print(alpha_v)
        polyreg=linear_model.Lasso(alpha=alpha_v)
        polyreg.fit(x_modified_train,y_train)
        train_err.append(sqrt(mean_squared_error(y_train, polyreg.predict(x_modified_train))))
        test_err.append(sqrt(mean_squared_error(y_test, polyreg.predict(x_modified_test))))
    
    #print(test_err)
    min_Test_error=min(test_err)
    Lambda_min_test_error=(test_err.index(min(test_err))+1)/20
    print(f"""Best lambda value is {Lambda_min_test_error} as it has the lowest test error of all lambadas: {min_Test_error: 3.5f}""")
    #Caculate RSME without Lasso for ploynoial of degree=10
    model7= linear_model.LinearRegression()
    model7.fit(x_modified_train, y_train)
    y_test7=model7.predict(x_modified_test)
    linear_rmse=sqrt(mean_squared_error(y_test, y_test7))
    print('Test RMSE  with Normal linear Regression is :',linear_rmse)
    print('Impoverment in model by using lasso(lambda=1) over normal Linear regression for ploynomial degree of 10 is : ',min_Test_error-linear_rmse)
    plt.title('Lasso')
    plt.xlabel('Alpha value')
    plt.ylabel('RMSE')
    plt.plot(np.linspace(0,1,20),train_err,'bo-',label='Train')
    plt.plot(np.linspace(0,1,20),test_err,'ro-',label='Test')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    lasso_reg()