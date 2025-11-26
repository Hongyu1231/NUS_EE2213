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
c3_init = x3.copy()

centers_init = np.array([c1_init, c2_init, c3_init])
# print(centers_init)

# Parameters to be changed in exam
fuzzier = 2
max_iterations = 100

# Assignment Step: Fix centers, update membership
def update_membership(data_points, centers, fuzzier):
    '''
    Parameters:
        datapoints: ndarray of shape (n_samples, n_features)
        centers: ndarray of shape (n_clusters, n_features)
        fuzzier: fuzzifier ([1.25,2])
        
    Returns:
        W: ndarray of shape (n_samples, n_clusters)
    '''
    n_samples = data_points.shape[0]
    n_clusters = centers.shape[0]
    W = np.zeros((n_samples, n_clusters)) # initialize membership matrix

    for i in range(n_samples):
        for k in range(n_clusters):
            denom = 0.0 # Denominator for membership calculation

            # Calculate ||x_i - c_k||
            dist_k = np.linalg.norm(data_points[i] - centers[k]) + 1e-10  # Avoid division by zero

            for j in range(n_clusters):
                # Calculate ||x_i - c_j||
                dist_j = np.linalg.norm(data_points[i] - centers[j]) + 1e-10
    
                ratio = (dist_k / dist_j)
                denom += ratio ** (2 / (fuzzier - 1))
            W[i, k] = 1 / denom
    return W

# Centroid Update Step: Fix membership, update centers
def update_centers(data_points, W, fuzzier):
    '''
    Parameters:
        datapoints: ndarray of shape (n_samples, n_features)
        W: ndarray of shape (n_samples, n_clusters)
        fuzzier: fuzzifier ([1.25,2])
        
    Returns:
        centers: ndarray of shape (n_clusters, n_features)
    '''
    n_clusters = W.shape[1]
    centers = np.zeros((n_clusters, data_points.shape[1]))

    for k in range(n_clusters):
        numerator = data_points.T @ (W[:, k] ** fuzzier)
        denominator = np.sum(W[:, k] ** fuzzier)
        centers[k] = numerator / denominator
    return centers

# Fuzzy_Cmeans Clustering
def fuzzy_Cmeans(data_points, center_init, fuzzier, max_iterations, tol=1e-4):
    centers = center_init.copy()
    for _ in range(max_iterations): 
        W = update_membership(data_points, centers, fuzzier)
        new_centers = update_centers(data_points, W, fuzzier)
        if np.linalg.norm(new_centers - centers) < tol:
            break
        centers = new_centers
    return centers, W

centers, W = fuzzy_Cmeans(data_points, centers_init, fuzzier, max_iterations)
print("Converged centers :", centers)
print("Final membership matrix (W):", W)
