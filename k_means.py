#Q1: Basic K-Means Clustering
#📊 Problem

#Create 2 clusters from dataset.

import numpy as np
from sklearn.cluster import KMeans
X = np.array([
    [1],
    [2],
    [3],
    [10],
    [11],
    [12]
])
model=KMeans(
    n_clusters=2,
    random_state=42

)
model.fit(X)
print("cluster centeriod:",model.cluster_centers_)
print("labels:",model.labels_)

#Q2: Predict Cluster of New Point
#📊 Problem
#Predict cluster for:
#x=5
import numpy as np
from sklearn.cluster import KMeans
X = np.array([
    [1],
    [2],
    [3],
    [10],
    [11],
    [12]
])
model=KMeans(
    n_clusters=2,
    random_state=42
)
model.fit(X)
print("prediction for 5:",model.predict([[5]]))

#Q3: 2D K-Means Clustering
#📊 Problem

#Cluster 2D points.

import numpy as np
from sklearn.cluster import KMeans
X = np.array([
    [1,2],
    [2,3],
    [3,4],
    [10,11],
    [11,12],
    [12,13]
])
model=KMeans(
    n_clusters=2,
    random_state=42
)
model.fit(X)
print("cluster centroid:",model.cluster_centers_)
print("labels:",model.labels_)

#Q4: Visualize Clusters
#📊 Problem
#Plot clusters using matplotlib.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
X = np.array([
    [1,2],
    [2,3],
    [3,4],
    [10,11],
    [11,12],
    [12,13]
])
model=KMeans(
    n_clusters=2,
    random_state=42
)
model.fit(X)
plt.scatter(X[:,0],X[:,1],c=model.labels_,cmap='viridis')
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],c='red',marker='x',s=200)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

#Q5: Elbow Method
#📊 Problem
#Find best K using Elbow Method.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
X = np.array([
    [1,2],
    [2,3],
    [3,4],
    [10,11],
    [11,12],
    [12,13]
])
wcss=[]
for k in range(1,6):
    model=KMeans(
        n_clusters=2,
        random_state=42

    )
    model.fit(X)
    wcss.append(model.inertia_)
plt.plot(range(1,6), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

#Q6: Silhouette Score
#📊 Problem
#Evaluate cluster quality.
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
X = np.array([
    [1,2],
    [2,3],
    [3,4],
    [10,11],
    [11,12],
    [12,13]
])
model=KMeans(
    n_clusters=2,
    random_state=42
)
model.fit(X)
score=silhouette_score(X,model.labels_)
print("Silhouette Score:", score)

#Q7: Compare Different K Values
#📊 Problem
#Find silhouette score for:
#K=2
#K=3
#K=4
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
X = np.array([
    [1,2],
    [2,3],
    [3,4],
    [10,11],
    [11,12],
    [12,13]
])
for k in range(2, 5):
    model=KMeans(
        n_clusters=k,
        random_state=42
    )
    model.fit(X)
    score=silhouette_score(X,model.labels_)
    print(f"Silhouette Score for K={k}:", score)

#Q8: K-Means on Real Dataset
#📊 Problem

#Cluster Iris dataset.
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
# Dataset
iris = load_iris()
X = iris.data
# Model
model = KMeans(
    n_clusters=3,
    random_state=42
)
model.fit(X)
print("Labels:",model.labels_)
#
#Q9: Cluster Visualization using PCA
#📊 Problem
#Reduce dimensions and visualize clusters.
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

iris = load_iris()

X = iris.data

# PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

# KMeans
model = KMeans(
    n_clusters=3,
    random_state=42
)

labels = model.fit_predict(X_pca)

# Plot
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=labels
)

plt.show()