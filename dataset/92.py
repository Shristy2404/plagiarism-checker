import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Creation of Dataframe using import
dataset_train = pd.read_csv('project - part D - training data set.csv',usecols=list(range(1,3)))
dataset_test =pd.read_csv('project - part D - testing data set.csv',usecols=list(range(1,3)))

#defining independent and dependent features
x_train= dataset_train.drop(columns = 'Son',axis =1)
y_train = dataset_train.Son
x_test =dataset_test.drop(columns ='Son',axis =1)
y_test = dataset_test.Son

#Defining the polynomial features ,the polynomial definition,model fit with the Lasso for training data and
# evaluation for train and test data
def poly_fit_lasso(x_train_data,y_train_data,x_test_data,y_test_data,poly_deg =1):
    poly_obj = PolynomialFeatures(poly_deg) # Passing the required polydegree to have the feature preparation
    
    #Transforming the feature to the poly object using the fit_transform
    x_data_train_transform = poly_obj.fit_transform(x_train_data) # Training data poly transform
    x_data_test_transform = poly_obj.fit_transform(x_test_data) # Testing data poly transform
    
    #Fit training data using LASSO wit hthe default alpha
    model_cbf_lasso = Lasso().fit(x_data_train_transform,y_train_data)
    model_coeff_lasso = model_cbf_lasso.coef_
    model_intercept_lasso =model_cbf_lasso.intercept_
    #Predicting using model for training data
    y_predict_train_lasso = model_cbf_lasso.predict(x_data_train_transform)
    
    #Predicting using model for the test data
    y_predict_test_lasso = model_cbf_lasso.predict(x_data_test_transform)
    
    #Evaluation of Training data
    RMSE_train = math.sqrt(mean_squared_error(y_train_data,y_predict_train_lasso))
    R2_train = r2_score(y_train_data,y_predict_train_lasso)
    
    #Evaluation of Testing data
    RMSE_test = math.sqrt(mean_squared_error(y_test_data,y_predict_test_lasso))
    R2_test = r2_score(y_test_data,y_predict_test_lasso)
    
    return RMSE_train,R2_train,RMSE_test,R2_test,model_coeff_lasso,model_intercept_lasso,y_predict_train_lasso,x_train_data.values

#Function to plot dictionary objects
def plot_data_dict(dictionary): # Argument:dictionary object
    dict_list = list(dictionary.items())
    x,y = zip(*dict_list)
    return x,y


#defining of dictionary to collect the rmse & R2 of train
RMSE_train_lasso = dict()
R2_train_lasso = dict()

#defining of the dictionary to collect the rmse & r2 of the test
RMSE_test_lasso = dict()
R2_test_lasso = dict()

#defining of the dictionary to collect the coeff and intercept from the lasso
coeff_train_lasso = dict()
intercept_train_lasso =dict()

#defining the dictionary Y-presdict from the training data
y_predict_training_lasso=dict() 
x_data_training_lasso =dict() # X training in dictionary
# fitting the Lasso Regression for polynom from 1 to 10 : purporse to compare with the ordinary polynom fit
    # in the poly exercise: Expectation the Lasso would make sparse coeff and theree by high chance that 
    # RMSE would be decreased on test data
for i in range(1,11):
    lasso_reg_poly = poly_fit_lasso(x_train[['Father']],y_train,x_test[['Father']],y_test,i)
    RMSE_train_lasso[i]=(lasso_reg_poly[0])
    R2_train_lasso[i] =(lasso_reg_poly[1])
    RMSE_test_lasso[i]= (lasso_reg_poly[2])
    R2_test_lasso[i]= (lasso_reg_poly[3])
    lasso_reg_poly_coeff=lasso_reg_poly[4][1:]# due to poly feature class
    coeff_train_lasso[i]=(lasso_reg_poly_coeff)
    intercept_train_lasso[i]=(lasso_reg_poly[5])
    y_predict_training_lasso[i]=(lasso_reg_poly[6])
    x_data_training_lasso[i]=(lasso_reg_poly[7])
    if i ==10:
        print ('For polynomial of degree 10 _Lasso Regression RMSE as follows:')
        print ('RMSE for Training data by Lasso Regression :',RMSE_train_lasso[i])
        print ('RMSE for Testing data by Lasso Regression :',RMSE_test_lasso[i])


#Plotting the RMSE for Train and test for Lasso
plt.figure()
plt.title('Lasso_fit_RMSE')
plt.xlabel('Degree of poly fit with Lasso')
plt.ylabel('RMSE_Train & Test')
plt.xticks(np.arange(0,11,1))
plt.plot(*plot_data_dict(RMSE_train_lasso),marker ='*',label ='Training_Set')
plt.plot(*plot_data_dict(RMSE_test_lasso),marker ='o',label ='Test_Set')
plt.savefig('Polynomial_Regression_Lasso_RMSE.png',format='png',quality =95)
plt.show

#Plotting the Lasso Regression curve for Train
plt.figure()
for keys in x_data_training_lasso.keys():
    x_temp= x_data_training_lasso.get(keys).ravel()
    y_temp = y_predict_training_lasso.get(keys)
    sorting_x_y=sorted(zip(x_temp,y_temp),key=lambda x: [x[0]])
    
    #plt.scatter(x_train,y_train,color ='b',marker ='o',label = 'Train set')
    plt.plot(*zip(*sorting_x_y),label = 'poly_fit_'+str(keys))
    if keys ==10:
            x_temp_l_10= x_data_training_lasso.get(keys).ravel()
            y_temp_l_10 = y_predict_training_lasso.get(keys)
            sorting_x_y_l_10=sorted(zip(x_temp_l_10,y_temp_l_10),key=lambda x: [x[0]])
            plt.plot(*zip(*sorting_x_y_l_10),label = 'poly_fit_'+str(keys))
plt.xticks(np.linspace(60,80,11))
plt.yticks(np.linspace(60,80,11))
plt.xlabel('Fathers Height')
plt.ylabel('Sons Height')
plt.title('Lasso_fit')
plt.legend(loc = 'lower right',fontsize =8)
plt.scatter(x_train[['Father']],y_train)
plt.savefig('Polynomial_Regression_Lasso.png',format='png',quality =95)
plt.show()
