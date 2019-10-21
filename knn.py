import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

class KNNClassifier:

    def __init__(self,n_neighbors=3):
        """
        Args: n_neighbors - Nearest neighbors considered while making a prediction, defaults to 3.
        """
        self.n_neighbors = n_neighbors
    
    def train(self,X_train,y_train):
        """
        Args: X_train (Features), y_train (Labels)
        This function trains the model....Too obvio?
        !!!CAUTION: Make sure that inputs are pandas.DataFrame objects of same length!!!
        """
        assert X_train.shape[0]==y_train.shape[0]
        self.X = X_train
        self.y = y_train

    def process_probable_predictions(self,probable_predictions):
        """
        Processes probable predictions recieved from the predict function.
        """
        assert self.n_neighbors==len(probable_predictions)
        initial_prediction = probable_predictions[0]
        pred_dict = {}
        for i in range(self.n_neighbors):
            pred_dict[probable_predictions[i]] = pred_dict.get(probable_predictions[i], 0) + 1
        
        majority = max(pred_dict,key=pred_dict.get)
        if pred_dict[majority]>1:
            return majority
        return initial_prediction

    def predict(self,x):
        """
        Args: x, numpy array of query code features
        Returns: prediction - Prediction based on preferntial voting.
                ranked_list - Sorted list of most similar source codes.
        """
        scores = []
        indices = []
        for index,row in self.X.iterrows():
            diff_vec = np.array(x) - np.array(row)
            scores.append(np.sqrt(diff_vec.dot(diff_vec)))
            indices.append(index)
        
        sorted_list = sorted(zip(scores,indices))
        
        probable_predictions = []
        for i in range(self.n_neighbors):
            probable_predictions.append(self.y[sorted_list[i][1]])

        return self.process_probable_predictions(probable_predictions),sorted_list

    def batch_predictions(self,X_test):
        """
        Args: X_test - Batch of test dataset of shape (batch_size,num_features)
        Returns: predictions - numpy.array object with prediction for each source codes shape: (batch_size,)
        """
        y_preds = []
        for index,row in X_test.iterrows():
            pred,ranked = self.predict(np.array(row))
            y_preds.append(pred)
        return y_preds

    def score(self,X_test,y_test):
        """
        Args: X_test - Batch of test dataset of shape (batch_size,num_features)
            y_test - Batch of test labels of shape (batch_size,1) or (batch_size,)
        Returns: accuracy_score - Accuracy of the model
        !!!CAUTION: MAKE SURE THAT X_test is pandas.DataFrame object or the door to Narnia will be opened!!!
        """
        y_preds = self.batch_predictions(X_test)
        return accuracy_score(y_test,y_preds)