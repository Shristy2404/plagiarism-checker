#!/usr/bin/env python
# coding: utf-8

# In[42]:


#!/usr/bin/env python
# coding: utf-8

# Importing the needed files

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math 

# the library we will use to create the model 
from sklearn import linear_model 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.preprocessing import PolynomialFeatures

# Step 2 - Reading the dataset  

dataset = pd.read_csv(r'C:\Users\Thamizh\Desktop\ammu\AIML\Materials\Project - Part D - C1 (Regression) - PGP in AI ML\project - part D - training data set.csv',delimiter=',')
ds = pd.read_csv(r'C:\Users\Thamizh\Desktop\ammu\AIML\Materials\Project - Part D - C1 (Regression) - PGP in AI ML\project - part D - testing data set.csv',delimiter=',')

# Reshaping values

xTrain = dataset['Father'].values.reshape(-1,1)
yTrain = dataset['Son'].values.reshape(-1,1)
xTest = ds['Father'].values.reshape(-1,1)
yTest = ds['Son'].values.reshape(-1,1)

#lasso reg
poly = PolynomialFeatures(degree=10)
xTrain_modified = poly.fit_transform(xTrain)
xTest_modified =poly.fit_transform(xTest)
lassoreg=linear_model.Lasso()
lassoreg.fit(xTrain_modified,yTrain)
print("RMSE for test 10th degree poly ",math.sqrt(mean_squared_error(yTrain,lassoreg.predict(xTrain_modified))))
print("RMSE for test 10th degree poly ",math.sqrt(mean_squared_error(yTest,lassoreg.predict(xTest_modified))))


trainerr =[]
testerr=[]
alpha_v = np.linspace(0,1,10)
for alpha_i in alpha_v:
    reg=linear_model.Lasso(alpha=alpha_i)
    reg.fit(xTrain,yTrain)
    trainerr.append(math.sqrt(mean_squared_error(yTrain,reg.predict(xTrain))))
    testerr.append(math.sqrt(mean_squared_error(yTest,reg.predict(xTest))))

print("Train RMSE",trainerr,'\n')
print("Test RMSE",testerr)
plt.title('Lasso')
plt.xlabel('Alpha value')
plt.ylabel('RMSE')
plt.plot(np.linspace(0,1,10),trainerr,'bo-',label='Train')
plt.plot(np.linspace(0,1,10),testerr,'ro-',label='Test')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




