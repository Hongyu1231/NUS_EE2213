import numpy as np
import cvxpy as cp


# Please replace "StudentMatriculationNumber" with your actual matric number in the filename
# Please do NOT change the function names in this file.
# Filename should be: A3_StudentMatriculationNumber.py (replace “StudentMatriculationNumber” with your own your student matriculation number).

def optimize_shipments(supply, demand, cost_matrix):
    """
    Problem 2: Logistics Optimization
    
    Inputs:
    :supply: list of int
        List of factory capacities [China, India, Brazil]
    :demand: list of int
        List of market demands [Singapore, US, Germany, Japan]
    :cost_matrix: 2D list (3x4)
        3x4 matrix where cost_matrix[i][j] is cost from factory i to market j
        Rows correspond to factories [China, India, Brazil].
        Columns correspond to markets [Singapore, US, Germany, Japan].
        
    Returns:
    :minimal_cost: float
        The total minimized transportation cost.
    :shipment_matrix: numpy.ndarray
        3x4 array of integers where shipment_matrix[i, j] is units 
        shipped from factory i to market j.
        Rows correspond to factories [China, India, Brazil].
        Columns correspond to markets [Singapore, US, Germany, Japan].
    """

    # Your ILP formulation and solution code goes here
    
    supply = np.asarray(supply, dtype=float)
    demand = np.asarray(demand, dtype=float)
    C = np.asarray(cost_matrix, dtype=float)

    X = cp.Variable((3, 4), integer=True)

    constriants = [
        X >= 0,
        cp.sum(X, axis=1) <= supply,
        cp.sum(X, axis=0) == demand,
    ]

    objective = cp.Minimize(cp.sum(cp.multiply(C, X)))

    prob = cp.Problem(objective, constriants)

    prob.solve()

    shipment_matrix = np.rint(X.value).astype(int)
    minimal_cost = float(np.sum(C * shipment_matrix))
    # Replace the following with actual return values
    return minimal_cost, shipment_matrix


def gradient_descent(learning_rate, num_iters):
    """
    Problem 2: Gradient Descent

    Inputs:
    :learning_rate: float
        The learning rate for gradient descent. Value between 0 and 0.2.
    :num_iters: int
        Number of gradient descent iterations.

    Returns:
    :w_out: numpy.ndarray
        Array of length num_iters containing updated w values at each step.
    :f_out: numpy.ndarray
        Array of length num_iters containing f(w) = 1 + (w - 5)^2 at each step.
    """

    # Initialization
    w = 3.5
    w_out = np.zeros(num_iters)
    f_out = np.zeros(num_iters)

    # Your gradient descent code goes here
    for t in range(num_iters):
        grad = 2 * (w - 5) 
        w = w - learning_rate * grad
        w_out[t] = w
        f_out[t] = 1 + (w - 5) ** 2

    # Replace the following with actual return values
    return w_out, f_out
