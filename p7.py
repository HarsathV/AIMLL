import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
iris=datasets.load_iris()
x=pd.DataFrame(iris.data)
x.columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y=pd.DataFrame(iris.target)
y.columns=['Targets']
kmeans=KMeans (n_clusters=3)
clusters=kmeans.fit_predict(x)
from scipy.stats import mode
labels=np.zeros_like(clusters)
for i in range(3):
    cat= (clusters==i)
    labels[cat]=mode(iris.target[cat])[0]
acc=accuracy_score(iris.target,labels)
print('Accuracy = ',acc)
plt.figure(figsize=(10,10))
colormap=np.array(['red','lime','blue'])
plt.subplot(2,2,1)
plt.scatter(x.Petal_Length,x.Petal_Width,c=colormap[y.Targets],s=40)
plt.title('Real Clusters')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.subplot(2,2,2)
plt.scatter(x.Petal_Length,x.Petal_Width,c=colormap[labels],s=40)
plt.title('Kmeans Clusters')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
from sklearn import preprocessing
scaler=preprocessing.StandardScaler()
scaler.fit(x)
scaled_x=scaler.transform(x)
xs=pd.DataFrame(scaled_x,columns=x.columns)
from sklearn.mixture import GaussianMixture
gmm=GaussianMixture(n_components=3)
gmm_y=gmm.fit_predict(xs)
labels=np.zeros_like(clusters)
for i in range(3):
    cat=(gmm_y==i)
    labels[cat]=mode(iris.target[cat])[0]
acc=accuracy_score(iris.target,labels)
print('Accuracy using gmm= ',acc)
plt.subplot(2,2,3)
plt.scatter(x.Petal_Length,x.Petal_Width,c=colormap[gmm_y],s=40)
plt.subplots_adjust(hspace=0.4,wspace=0.4)
plt.title('gmm Clusters')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')   
plt.show() 




