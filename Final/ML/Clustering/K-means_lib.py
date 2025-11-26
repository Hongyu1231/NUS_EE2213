import numpy as np

# Coordinates of data points, to be changed in the exam
x1 = np.array([0, 0])
x2 = np.array([1, 0])
x3 = np.array([0, 1])
x4 = np.array([1, 1])

data_points = np.array([x1, x2, x3, x4])
# print(data_points)

# Initial centers
c1_init = x1.copy()
c2_init = x2.copy()
# The initial centers
centers_init = np.array([c1_init, c2_init])
# For debug purposes
# print(centers_init)

# Parameters to be changed in exam
max_iterations = 100
# Need to be consistent with centre_init
num_clusters = 2

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=num_clusters, init=centers_init, max_iter=max_iterations, n_init=1)
#n_init: The number of times the KMeans algorithm will run with different centroid seeds
#        Setting n_init=1 means it will only run once, using the given centers_init
kmeans.fit(data_points)
print("KMeasn centers from sklearn:", kmeans.cluster_centers_)
print("KMeasn labels from sklearn:", kmeans.labels_)
