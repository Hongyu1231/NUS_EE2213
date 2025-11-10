import numpy as np


# Please replace "MatricNumber" with your actual matric number here and in the filename
def A1_A0311920U(x, y):
    """
    Input type
    :x type: numpy.ndarray
    :y type: numpy.ndarray

    Return type
    :euclidean_dist type: numpy.ndarray
    :manhattan_dist type: numpy.ndarray
   
    """

    # your code goes here
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Both inputs must be 1D NumPy arrays.")
    
    if len(x) != len(y):
        raise ValueError("Both input arrays must have the same length.")
    
    euclidean_dist = np.sqrt(np.sum((x - y)**2))
    manhattan_dist = np.sum(np.abs(x - y))
    euclidean_dist = np.array([np.round(euclidean_dist, 2)])
    manhattan_dist = np.array([np.round(manhattan_dist, 2)])

    # return in this order
    return euclidean_dist, manhattan_dist


