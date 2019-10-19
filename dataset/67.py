import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

traindata = pd.read_csv("training_data set.csv")
testdata = pd.read_csv("testing_data set.csv")

x_train = traindata['Father'].values.reshape(-1, 1)
y_train = traindata['Son'].values.reshape(-1, 1)
x_test = testdata['Father'].values.reshape(-1, 1)
y_test = testdata['Son'].values.reshape(-1, 1)

lassoreg = PolynomialFeatures(degree=10)
x_modified_train = lassoreg.fit_transform(x_train)
x_modified_test = lassoreg.fit_transform(x_test)
lassomodel= Lasso(alpha=0.5)
lassomodel.fit(x_modified_train, y_train)
y_predicted_test=lassomodel.predict(x_modified_test)
y_predicted_train=lassomodel.predict(x_modified_train)
print('Lasso RMSE Train:', sqrt(mean_squared_error(y_train, y_predicted_train)))
print('Lasso RMSE Test:', sqrt(mean_squared_error(y_test, y_predicted_test)))

train_err = []
test_err = []
alpha_vals=np.linspace(0, 1, 10)
for alpha_v in alpha_vals:
    polyreg10 = Lasso(alpha=alpha_v)
    polyreg10.fit(x_modified_train, y_train)
    train_err.append(sqrt(mean_squared_error(y_train, polyreg10.predict(x_modified_train))))
    test_err.append(sqrt(mean_squared_error(y_test, polyreg10.predict(x_modified_test))))

plt.title('Lasso Regression ')
plt.xlabel('Alpha value')
plt.ylabel('RMSE')
plt.plot(np.linspace(0, 1, 10), train_err, 'bo-', label='Train')
plt.plot(np.linspace(0, 1, 10), test_err, 'ro-', label='Test')
plt.legend()
plt.show()
