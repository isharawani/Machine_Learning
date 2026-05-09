#Q1: Basic Hierarchical Clustering
#📊 Problem
#Cluster points into:
#👉 2 clusters
#Dataset:
#[1],[2],[10],[11]

import numpy as np
from sklearn.cluster import AgglomerativeClustering
X = np.array([
    [1],
    [2],
    [10],
    [11]
])
model=AgglomerativeClustering(
    n_clusters=2
)
model.fit(X)
print("labels:",model.labels_)


#Q2: 2D Hierarchical Clustering
#📊 Problem
#Cluster 2D points.
import numpy as np
from sklearn.cluster import AgglomerativeClustering
X = np.array([
    [1,2],
    [2,3],
    [10,11],
    [11,12]
])
model=AgglomerativeClustering(
    n_clusters=2
)
model.fit(X)
print("labels:",model.labels_)

#Q3: Dendrogram Visualization
#📊 Problem
#Visualize hierarchical clustering tree
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, single
X = np.array([
    [1],
    [2],
    [10],
    [11]
])
linked = linkage(X, method='ward')
dendrogram(linked)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

#Q4: Single Linkage Clustering
#📊 Problem
#Perform clustering using:
#👉 single linkage
import numpy as np
from sklearn.cluster import AgglomerativeClustering
X = np.array([
    [1],
    [2],
    [10],
    [11]
])
model=AgglomerativeClustering(
    n_clusters=2,
    linkage="single"
)
model.fit(X)
print("labels:",model.labels_)

#Q5: Complete Linkage Clustering
#📊 Problem#
#Use:
#👉 complete linkage
import numpy as np
from sklearn.cluster import AgglomerativeClustering
X = np.array([
    [1],
    [2],
    [10],
    [11]
])
model=AgglomerativeClustering(
    n_clusters=2,
    linkage="complete"
)
model.fit(X)
print("labels:",model.labels_)

#Q6: Average Linkage
#📊 Problem
#Use:
#👉 average linkage
import numpy as np

from sklearn.cluster import AgglomerativeClustering

X = np.array([
    [1],
    [2],
    [10],
    [11]
])

model = AgglomerativeClustering(
    n_clusters=2,
    linkage='average'
)
model.fit(X)
print("labels:", model.labels_)

#Q7: Visualize Clusters
#📊 Problem
#Plot hierarchical clusters.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
X = np.array([
    [1,2],
    [2,3],
    [10,11],
    [11,12]
])
model = AgglomerativeClustering(
    n_clusters=2
)
labels = model.fit_predict(X)
# Plot
plt.scatter(
    X[:,0],
    X[:,1],
    c=labels
)
plt.show()


#Q8: Real Dataset Problem
#📊 Problem
#Apply hierarchical clustering on Iris dataset.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data
model=AgglomerativeClustering(
    n_clusters=3
)
model.fit(X)
print("labels:",model.labels_)

#Q9: Compare Linkage Methods
#📊 Problem
#Compare:
#single
#complete
#average
#linkage methods.

import numpy as np
from sklearn.cluster import AgglomerativeClustering
X = np.array([
    [1],
    [2],
    [10],
    [11]
])
for method in ["single", "complete", "average"]:
    model=AgglomerativeClustering(
        n_clusters=2,
        linkage=method,
    )
    model.fit(X)
    print(f"labels (linkage={method}):", model.labels_)
