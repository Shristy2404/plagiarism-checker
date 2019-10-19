
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math


# In[2]:


train_df=pd.read_csv("project - part D - training data set.csv")


# In[3]:


test_df=pd.read_csv("project - part D - testing data set.csv")


# In[4]:


test_df


# In[5]:


X_train=train_df['Father']


# In[6]:


X_train=X_train.values.reshape(-1,1)


# In[7]:


y_train=train_df['Son']


# In[8]:


X_test=test_df['Father']


# In[9]:


X_test=X_test.values.reshape(-1,1)


# In[10]:


y_test=test_df['Son']
#y_test


# In[11]:


poly_lasso=PolynomialFeatures(degree=10)
modified_X_train_lasso=poly_lasso.fit_transform(X_train)
modified_X_test_lasso=poly_lasso.fit_transform(X_test)
reg_lasso=Lasso(alpha=0.5)
reg_lasso.fit(X_train,y_train)


# In[12]:


alpha_vals=np.linspace(0,1,10)


# In[13]:


train_err=[]
test_err=[]


# In[14]:


for alphaval in alpha_vals:
    lassoreg=Lasso(alpha=alphaval)
    lassoreg.fit(X_train,y_train)
    train_err.append(math.sqrt(mean_squared_error(y_train,lassoreg.predict(X_train))))
    test_err.append(math.sqrt(mean_squared_error(y_test,lassoreg.predict(X_test))))


# In[15]:


print("RMSE values for Training Error: ",train_err)
print("RMSE values for Testing Error: ",test_err)


# In[16]:


plt.ylabel('Lasso Regression')
plt.xlabel('Alpha values')
plt.ylabel('RMSE')
plt.plot(np.linspace(0,1,10),train_err,'bo-',label='Train')
plt.plot(np.linspace(0,1,10),test_err,'ro-',label='Test')
plt.legend()
plt.show()


# In[17]:


nl=np.linspace(0,1,10)


# In[18]:


print("Alpha value for minimum test error: ", nl[test_err.index(min(test_err))])


# In[19]:


print("Minimum test error: ", min(test_err))

