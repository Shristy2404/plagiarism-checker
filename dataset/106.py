import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

train_err = []
test_err = []
dataset_train = pd.read_csv('project - part D - training data set.csv')
X_train = dataset_train['Father'].values.reshape(-1,1)
y_train = dataset_train['Son'].values.reshape(-1,1)

dataset_test = pd.read_csv('project - part D - testing data set.csv')
X_test = dataset_test['Father'].values.reshape(-1,1)
y_test = dataset_test['Son'].values.reshape(-1,1)

poly = PolynomialFeatures(degree = i)
modified_X = poly.fit_transform(X_train)
modified_X_test = poly.fit_transform(X_test)

alpha_vals = np.linspace(0,1,10)
for alpha_v in alpha_vals:
    reg = Lasso(alpha=alpha_v)
    reg.fit(modified_X, y_train)

    train_err.append(sqrt(mean_squared_error(y_train, reg.predict(modified_X))))
    test_err.append(sqrt(mean_squared_error(y_test, reg.predict(modified_X_test))))

plt.title('Lasso')
plt.plot(np.linspace(0,1,10),test_err, color = 'red', label = 'Test')
plt.plot(np.linspace(0,1,10),train_err, color = 'blue', label = 'Train')
plt.xlabel("Alpha Values")
plt.ylabel("RMSE")
plt.legend()
plt.show()