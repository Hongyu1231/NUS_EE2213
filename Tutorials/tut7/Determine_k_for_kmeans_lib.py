import numpy as np
from sklearn.datasets import load_wine

# load the wine dataset as a dictionary-like object
wine = load_wine()
print("Wine dataset keys:", wine.keys())
print("Wine dataset description:", wine.DESCR) # Description of the dataset

X = wine.data
y = wine.target

# print("Wine dataset feature matrix:", X)
# print("Wine dataset target vector:", y)

from sklearn.cluster import KMeans

J = {}
np.random.seed(42)  # For reproducibility
for n_clusters in range(2, 11):
    centers_init = X[np.random.choice(X.shape[0], n_clusters, replace = False)] # np.random.choice(): k random indices.
    kmeans = KMeans(n_clusters=n_clusters, init=centers_init, n_init=1)
    #n_init: The number of times the KMeans algorithm will run with different centroid seeds
    #        Setting n_init=1 means it will only run once, using the given centers_init
    kmeans.fit(X)
    within_cluster_var = np.sum((X - kmeans.cluster_centers_[kmeans.labels_]) ** 2)
    J[n_clusters] = within_cluster_var
    print(f"Within-cluster variance for {n_clusters} clusters:", within_cluster_var)

import matplotlib.pyplot as plt
plt.plot(list(J.keys()), list(J.values()), marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster variance')
plt.show()