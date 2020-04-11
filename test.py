from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

from knn import *
from sklearn.naive_bayes import ComplementNB
from datetime import datetime
import pandas as pd

# READ DATASET
df = pd.read_csv('final_dataset.csv')
df = df.drop(['file_name','llc_0','lwc_0','vocabulary','length','effort','volume','loc'],axis=1)
df = df.loc[:, df.isin([' ','NULL',0]).mean() < .6] # Drop if missing values > 60%
y = df.pop('author')
X = df.copy()

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

# MODEL
# start=datetime.now()

clf = KNNClassifier(n_neighbors=3)
clf.train(X_train,y_train)
y_preds = clf.batch_predictions(X_test)
print('\nClassification Report')
print(classification_report(y_test,y_preds))
print('Accuracy: ',clf.score(X_test,y_test))

clf2 = ComplementNB()
clf2.fit(X_train,y_train)
y_preds_2 = clf2.predict(X_test)
print('\nNaive Bayes Classification Report')
print(classification_report(y_test,y_preds_2))
print('NB Accuracy: ',clf2.score(X_test,y_test))

# print('Time taken:',datetime.now()-start)

# # CONFUSION MATRIX VISUALIZATION
# y_actu = pd.Series(np.array(y), name='Actual')
# y_pred = pd.Series(np.array(y_preds), name='Predicted')
# df_confusion = pd.crosstab(y_actu, y_pred)

# import matplotlib.pyplot as plt
# def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
#     plt.matshow(df_confusion, cmap=cmap) # imshow
#     #plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(df_confusion.columns))
#     plt.xticks(tick_marks, df_confusion.columns, rotation=45)
#     plt.yticks(tick_marks, df_confusion.index)
#     #plt.tight_layout()
#     plt.ylabel(df_confusion.index.name)
#     plt.xlabel(df_confusion.columns.name)
#     plt.savefig('confusion_matrix.png')

# plot_confusion_matrix(df_confusion)
# print('\nConfusion Matrix')
# print(confusion_matrix(y,y_preds))