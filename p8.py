import pandas as pd 
import numpy as np 
import sklearn as sl
from sklearn.model_selection import train_test_split 
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris=datasets.load_iris() 
iris.data.shape, iris.target.shape 
X_train,X_test,y_train,y_test=train_test_split(iris.data, iris.target, test_size=0.2,random_state=0)
X_train.shape, y_train.shape 
X_test.shape,y_train.shape 
clf=KNeighborsClassifier()
clf.fit(X_train,y_train)
KNeighborsClassifier (algorithm= 'auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=None, n_neighbors=5, p=2,
weights=' uniform')
clf.score (X_test,y_test)
accuracy=clf.score (X_test, y_test)
print(accuracy)
example_measures=np.array([[4.7, 3.2,2,0.2], [5.1, 2.4,4.3,1.3]]) 
example=example_measures.reshape(2,-1)
prediction=clf.predict(example)
print (prediction)