#Hierachical clustering

#import the libraries

import pandas as pd
import matplotlib.pyplot as plt
#import dataset
dataset=pd.read_csv("Mall_Customers.csv")
x=dataset.iloc[:,:].values
x[0]
array([15, 39], dtype=int64)
#dendogram to find optimal no.of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distance")
plt.show()

#train the model
from sklearn.cluster import AgglomerativeClustering
clustering=AgglomerativeClustering(n_clusters=5)
y_hc=clustering.fit_predict(x)
#visualising the clusters
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],c='red',label='Cluster1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],c='green',label='Cluster2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],c='pink',label='Cluster3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],c='blue',label='Cluster4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],c='orange',label='Cluster5')
plt.title("Cluster of Customers")
plt.xlabel("Annual income(k$)")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()
