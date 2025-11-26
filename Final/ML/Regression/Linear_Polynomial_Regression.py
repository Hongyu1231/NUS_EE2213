from sklearn.metrics import mean_squared_error
import numpy as np
from numpy.linalg import inv, matrix_rank
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


# Input data to be changed (No need to add augmented 1)
X = np.array([[45, 26, 3], [68, 19, 7], [32, 25, 5], [55, 35, 9], [28, 21, 2]]) # 3 features with 4 samples (The number of numbers in the inner paranthesis is the number of features)
Y = np.array([2, 5, 3, 7, 1])

# Test data to be changed
X_test = np.array([[30, 18, 4]])


# # Z-score normalization
# scaler = StandardScaler()
# X = scaler.fit_transform(X) # Scale to mean=0, variance=1
# X_test = scaler.fit_transform(X_test)


# Order to be changed
order = 2
poly = PolynomialFeatures(order) # Augmented 1 handled by this manually
X_poly = poly.fit_transform(X) 
# Uncomment for debug purposes
print("The preprocessed X:")
print(X)


def check_inverse_rank(matrix):
    rank = matrix_rank(matrix)
    print("Matrix rank is : " + str(rank))
    print("Matrix shape is : " + str(matrix.shape))

    if matrix.shape[0] == matrix.shape[1]:
        if matrix.shape[0] == rank:
            print("Matrix is invertible.")
        else:
            print("Matrix is not invertible.")
    else:
        print("Matrix is not square, so matrix isn't invertible.")

    return (matrix.shape[0] == matrix.shape[1]) and (matrix.shape[0] == rank)


check_inverse_rank(X_poly.T @ X_poly)


# The coefficient: [1, x1, x2, x3, x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2, .......]
w = inv(X_poly.T @ X_poly) @ X_poly.T @ Y
print("w is : " + str(w))

y_fit = X_poly @ w
MSE = mean_squared_error(Y, y_fit)
print("MSE is : " + str(MSE))


X_test_linear = poly.transform(X_test)
y_test = X_test_linear @ w
print("y_test is :" + str(y_test))



