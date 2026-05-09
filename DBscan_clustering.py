#Q1: Basic DBSCAN Clustering
#📊 Problem
#Cluster the following points:
#[1],[2],[3],[10]
#Use:
#eps = 2
#min_samples = 2
import numpy as np
from sklearn.cluster import DBSCAN
# Dataset
X = np.array([
    [1],
    [2],
    [3],
    [10]
])
# Model
model = DBSCAN(
    eps=2,
    min_samples=2
)
# Train
labels = model.fit_predict(X)
# Output
print("Cluster Labels:")
print(labels)

#Q2: 2D DBSCAN Clustering
#📊 Problem
#Cluster 2D points.
import numpy as np
from sklearn.cluster import DBSCAN
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [10, 11]
])
model = DBSCAN(
    eps=2,
    min_samples=2
)
labels = model.fit_predict(X)
model.fit(X)
print("Cluster Labels:",model.labels_)

#Q3: Visualize DBSCAN Clusters
#📊 Problem
#Plot clusters using matplotlib.#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
X = np.array([
    [1,2],
    [2,3],
    [10,11],
    [11,12]
])
model = DBSCAN(
    eps=2,
    min_samples=2
)
labels = model.fit_predict(X)
# Plot
plt.scatter(
    X[:,0],
    X[:,1],
    c=labels
)
plt.title("DBSCAN Clustering")
plt.show()
#
#Q4: Change eps Value
#📊 Problem
#Compare clustering for:
#eps = 1
#eps = 3
import  numpy as np
from sklearn.cluster import DBSCAN
X = np.array([
    [1],
    [2],
    [3],
    [10]
])
for eps_value in [1,3]:  #for min_pts_value in [2,3]:  
    model=DBSCAN(
        eps=eps_value,
        min_samples=2  #for min samples -> min_samples=min_pts
    )
    model.fit(X)
    print("eps:", eps_value)  #print("min_samples:",min_pts)
    print("labels:",model.labels_)

#Q6: Detect Outliers
#📊 Problem
#Find noise points.
import  numpy as np
from sklearn.cluster import DBSCAN
X = np.array([
    [1],
    [2],
    [3],
    [10]
])
model=DBSCAN(
    eps=2,
    min_samples=2
)
model.fit(X)
for i in range(len(X)):
    if model.labels_[i] == -1:
        print(f" Noise Point {X[i]} is an outlier.")

#Q7: DBSCAN on Real Dataset
#📊 Problem
#Apply DBSCAN on Iris dataset.#
import numpy as np
from sklearn.cluster import DBSCAN 
from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data
model=DBSCAN(
    eps=0.5,
    min_samples=5
)
model.fit(X)
print("labels:",model.labels_)

#Q8: Visualize Real Dataset Clusters
#📊 Problem
#Visualize DBSCAN clusters.
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data[:, :2]  # Use only the first two features for visualization
model=DBSCAN(
    eps=0.5,
    min_samples=5
)
model.fit(X)
plt.scatter(
    X[:,0],
    X[:,1],
    c=model.labels_
)
plt.title("DBSCAN Clustering on Iris Dataset")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()

#Q9: Compare K-Means vs DBSCAN
#📊 Problem
#Compare clustering outputs.#
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
X = np.array([
    [1,2],
    [2,2],
    [2,3],
    [8,8]
])
# KMeans
kmeans = KMeans(
    n_clusters=2,
    random_state=42
)
k_labels = kmeans.fit_predict(X)
print("KMeans:")
print(k_labels)
# DBSCAN
dbscan = DBSCAN(
    eps=2,
    min_samples=2
)
d_labels = dbscan.fit_predict(X)
print("DBSCAN:")
print(d_labels)