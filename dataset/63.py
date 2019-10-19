
# coding: utf-8

# In[46]:


## Step 1 - Importing the required libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# Remember what this line did?
get_ipython().run_line_magic('matplotlib', 'inline')
import math 

# the library we will use to create the model 
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn import metrics 


# In[47]:


# 2.1 Read the training dataset from the csv file using pandas and write into a pandas dataframe object named 'dataset'
dataset=pd.read_csv('D:\Sai_Gudipati\Personal\BITS_PILANI\Project\Project - Part D\\project - part D - training data set.csv')
X=dataset["Father"].values.reshape(-1,1)
Y=dataset["Son"].values.reshape(-1,1)


# In[48]:


# 2.2 Read the test dataset from the csv file using pandas and write into a pandas dataframe object named 'test_data'
test_data=pd.read_csv("D:\Sai_Gudipati\Personal\BITS_PILANI\Project\Project - Part D\project - part D - testing data set.csv")
X_test=test_data["Father"].values.reshape(-1,1)
Y_test=test_data["Son"].values.reshape(-1,1)


# In[49]:


poly=PolynomialFeatures(degree=10)
X_train=poly.fit_transform(X)
X_modified_test=poly.fit_transform(X_test)

train_err=[]
test_err=[]
alpha_vals=np.linspace(0,1,10)
i=0
for alpha_v in alpha_vals:
    reg=Lasso(alpha=alpha_v)
    reg.fit(X_train,Y)
    #print("Alpha=",alpha_v)
    #print("coefficient=",reg.coef_)
    train_rmse=math.sqrt(metrics.mean_squared_error(Y,reg.predict(X_train)))
    train_err.append(train_rmse)
    test_rmse=math.sqrt(metrics.mean_squared_error(Y_test,reg.predict(X_modified_test)))
    
    print("Alpha=",alpha_v)
    print("Train RMSE=",train_rmse)
    print("Test RMSE=",test_rmse)
    print("coeff",reg.coef_)

    test_err.append(test_rmse)
    
    if alpha_v==0:
        X_best_alpha=alpha_v
        X_best_rmse=test_rmse
    elif X_best_rmse > test_rmse:
        X_best_alpha = alpha_v
        X_best_rmse = test_rmse


print ("For the given set of data, best alpha is ", X_best_alpha)
print ("For the given set of data, optimal rmse= ", X_best_rmse)
plt.title("Lasso Polynomial")
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.plot(np.linspace(0,1,10),train_err,'bo-',label='Train')
plt.plot(np.linspace(0,1,10),test_err,'ro-',label='Test')
plt.legend()
plt.savefig('D:\Sai_Gudipati\Personal\BITS_PILANI\Project\Project - Part D\id_lasso_part_d.png')
plt.show()

