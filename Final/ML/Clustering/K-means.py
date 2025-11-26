import numpy as np


# The coordinates of data points. To be changed in exam
x1 = np.array([1, 2])
x2 = np.array([1.5, 1.8])
x3 = np.array([5, 8])
x4 = np.array([8, 8])
x5 = np.array([1, 0.6])
x6 = np.array([9, 11])

data_points = np.array([x1, x2, x3, x4, x5, x6])
# For debug purposes
# print(data_points) 

# To be changed in exam
# Initial centers 
c1_init = x1.copy()
c2_init = x4.copy()


centers_init = np.array([c1_init, c2_init])
# For debug purposes
# print(centers_init)

# Parameters to be changed in exam
max_iterations = 1000
# Need to be consistent with centre_init
num_clusters = 2


def k_means(data_points, centers_init, n_clusters, max_iterations, tol = 1e-4):
    centers = centers_init.copy()
    for _ in range(max_iterations):
        # Compute squared Euclidean distances to each centroid
        # Result shape: (n_samples, k)
        distances = np.linalg.norm(data_points[:, np.newaxis] - centers, axis = 2)

        # Assign each point to the index of the closest centroid
        closest_centroids = np.argmin(distances, axis=1) # Finds the index of the minimum value along each row

        # Update centroids to be the mean of the data points assigned to them
        new_centers = np.zeros((n_clusters, data_points.shape[1]))
        for i in range(n_clusters):
            new_centers[i] = data_points[closest_centroids == i].mean(axis=0)
            # print(np.linalg.norm(data_points[:, np.newaxis] - centers, axis=2))

        # End if centroids no longer change
        if np.linalg.norm(new_centers - centers) < tol:
            break
        centers = new_centers
    return centers, closest_centroids


# break step by step
# print(data_points[:, np.newaxis]) #add a new axis between the two existing ones.
# print(data_points[:, np.newaxis].shape)
# data_points[:, np.newaxis] - centers_init
# np.linalg.norm(data_points [:, np.newaxis] - centers_init, axis=2)

centers, labels = k_means(data_points, centers_init, num_clusters, max_iterations)
print("Converged centers :", centers)
print("cluster Labels :", labels)


from matplotlib import pyplot as plt
plt.scatter(data_points[:, 0], data_points[:, 1], c = labels, cmap = 'viridis', marker = 'o')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.title('K-Means Clustering')
plt.show()
                