from sklearn.model_selection import train_test_split
from knn import *

df = pd.read_csv('final_dataset.csv')
df = df.drop(['file_name','llc_0','lwc_0','vocabulary','length','effort','volume','loc'],axis=1)
y = df.pop('author')
X = df.copy()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
clf = KNNClassifier(n_neighbors=5)
clf.train(X_train,y_train)
print(clf.score(X_test,y_test))