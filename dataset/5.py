import pandas as pd 
import math 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn import metrics 
training_data = pd.read_csv('project - part D - training data set.csv')
x_train = training_data['Father'].values.reshape(-1,1)
y_train = training_data['Son'].values.reshape(-1,1)
test_data = pd.read_csv('project - part D - testing data set.csv')
x_test = test_data['Father'].values.reshape(-1,1)
y_test = test_data['Son'].values.reshape(-1,1)

poly = PolynomialFeatures(degree=10)
modified_x_train = poly.fit_transform(x_train)
modified_x_test = poly.fit_transform(x_test)
model= Lasso()
model.fit(modified_x_train,y_train)
y_train_predict = model.predict(modified_x_train)
training_mse = metrics.mean_squared_error(y_train, y_train_predict)
training_rmse = math.sqrt(training_mse)
y_test_predict = model.predict(modified_x_test)
testing_mse = metrics.mean_squared_error(y_test, y_test_predict)
testing_rmse = math.sqrt(testing_mse)
print('Training_RMSE: ',training_rmse)
print('Testing_RMSE: ',testing_rmse)
