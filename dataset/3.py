import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import scale
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error

d = pd.read_csv("project - part D - training data set.csv")
X = d["Father"].values.reshape(-1,1)
y = d["Son"].values.reshape(-1,1)

d = pd.read_csv("project - part D - testing data set.csv")
X1 = d["Father"].values.reshape(-1,1)
y1 = d["Son"].values.reshape(-1,1)

poly = PolynomialFeatures(degree=10)
modified_X = poly.fit_transform(X)
reg = Lasso()
reg.fit(modified_X, y)
print("Lasso Regression: Printing RMSE for Degree =10- Train RMSE:", math.sqrt(mean_squared_error(y, reg.predict(modified_X))))
modified_X1 = poly.fit_transform(X1)
print("Lasso Regression: Printing RMSE for Degree =10- Test RMSE:", math.sqrt(mean_squared_error(y1, reg.predict(modified_X1))))

#Lasso
train_err = []
test_err = []

# For varying alpha's and Polynominal Degree 10
alpha_vals = np.linspace(0,1,10)
for i in alpha_vals:
    reg = Lasso(alpha=i)
    reg.fit(modified_X, y)
    train_err.append(math.sqrt(mean_squared_error(y, reg.predict(modified_X))))
    print("Lasso Regression: Printing RMSE for Alpha =", i ,"- Train RMSE:",math.sqrt(mean_squared_error(y, reg.predict(modified_X))))
    test_err.append(math.sqrt(mean_squared_error(y1, reg.predict(modified_X1))))
    print("Lasso Regression: Printing RMSE for Alpha =", i ,"- Test RMSE:",math.sqrt(mean_squared_error(y1, reg.predict(modified_X1))))

plt.title('Lasso (Polynomial degree 10, Alpha varying 0 to 1)')
plt.xlabel('Alpha Value')
plt.ylabel('RMSE')
plt.plot(np.linspace(0,1,10), test_err, 'o-', label="Test", color='Blue')
plt.plot(np.linspace(0,1,10), train_err, 'o-',label="Train", color='Red')
plt.legend()
plt.show()

train_err = []
test_err = []
# For default alpha's and Polynominal Degree varying
for i in range(1, 11):
    poly = PolynomialFeatures(degree=i)
    modified_X = poly.fit_transform(X)
    reg = Lasso()
    reg.fit(modified_X, y)
    train_err.append(math.sqrt(mean_squared_error(y, reg.predict(modified_X))))
    print("Lasso Regression: Printing RMSE for Degree =", i ,"- Train RMSE:",math.sqrt(mean_squared_error(y, reg.predict(modified_X))))
    modified_X1 = poly.fit_transform(X1)
    test_err.append(math.sqrt(mean_squared_error(y1, reg.predict(modified_X1))))
    print("Lasso Regression: Printing RMSE for Degree =", i ,"- Test RMSE:",math.sqrt(mean_squared_error(y1, reg.predict(modified_X1))))

plt.title('Lasso (Polynomial degree varying 1 to 10, default alpha)')
plt.xlabel('Alpha Value')
plt.ylabel('RMSE')
plt.plot(np.linspace(0,1,10), test_err, 'o-', label="Test", color='Blue')
plt.plot(np.linspace(0,1,10), train_err, 'o-',label="Train", color='Red')
plt.legend()
plt.show()



