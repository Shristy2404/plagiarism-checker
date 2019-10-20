import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
class KNNClassifier:
    def __init__(self,n_neighbors=3):
        self.n_neighbors = n_neighbors
    
    """
    Arguments: X_train, y_train
    Both the variables must be DataFrame objects.
    """
    def train(self,X_train,y_train):
        self.X = X_train
        self.y = y_train

    def process_probable_predictions(self,probable_predictions):
        assert self.n_neighbors==len(probable_predictions)
        initial_prediction = probable_predictions[0]
        pred_dict = {}
        for i in range(self.n_neighbors):
            pred_dict[probable_predictions[i]] = pred_dict.get(probable_predictions[i], 0) + 1
        
        majority = max(pred_dict,key=pred_dict.get)
        if pred_dict[majority]>1:
            return majority
        return initial_prediction

    """
    Arguments: x, numpy array representation of the query
    """
    def predict(self,x):
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


    def score(self,X_test,y_test):
        y_preds = []
        for index,row in X_test.iterrows():
            pred,ranked = predict(np.array(row))
            y_preds.append(pred)
        return accuracy_score(y_test,y_preds)