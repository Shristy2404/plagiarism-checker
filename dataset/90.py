import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso

import warnings
warnings.filterwarnings('ignore')

train_data = pd.read_csv('project - part D - training data set.csv')
train_data = train_data.drop([train_data.columns[0]], axis=1)
x_train= train_data['Father'].values.reshape(-1, 1)
y_train = train_data['Son'].values.reshape(-1,1)

test_data = pd.read_csv('project - part D - testing data set.csv')
test_data = test_data.drop([test_data.columns[0]], axis=1)
x_test= test_data['Father'].values.reshape(-1, 1)
y_test = test_data['Son'].values.reshape(-1,1)

train_error =[]
test_error =[]

polynomial = PolynomialFeatures(degree=10)
modified_x_train = polynomial.fit_transform(x_train)
modified_x_test = polynomial.fit_transform(x_test)
lasso_regression = Lasso(max_iter=10e5)
lasso_regression.fit(modified_x_train, y_train)

print('Lasso Regression Train RMSE with default lambda with Degree 10:', math.sqrt(mean_squared_error(y_train, lasso_regression.predict(modified_x_train))))
print('Lasso Regression Test RMSE with default lambda with Degree 10:', math.sqrt(mean_squared_error(y_test, lasso_regression.predict(modified_x_test))))

train_error =[]
test_error =[]
alpha_values = np.linspace(0, 1, 10)
for alpha_v in alpha_values:
    lasso_regression = Lasso(alpha=alpha_v, max_iter=10e5)
    lasso_regression.fit(modified_x_train, y_train)
    train_error.append(math.sqrt(mean_squared_error(y_train, lasso_regression.predict(modified_x_train))))
    test_error.append(math.sqrt(mean_squared_error(y_test, lasso_regression.predict(modified_x_test))))

plt.title("Lasso Regression Plot")
plt.xlabel('Alpha Value')
plt.ylabel('RMSE')
plt.plot(np.linspace(0, 1, 10), train_error, 'bo-', label='Train')
plt.plot(np.linspace(0, 1, 10), test_error, 'ro-', label='Test')
plt.legend()
plt.show()
