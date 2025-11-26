from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, matrix_rank
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


# Input data to be changed (Must be 1 feature)
X = np.array([[1.6], [1.5], [1.0], [1.1], [1.2], [0.9]])
Y = np.array([3.1, 3.4, 2.9, 3.0, 3.8, 2.9])


# Test data to be changed
X_test = np.array([[0.5]])


# # Z-score normalization
# scaler = StandardScaler()
# X = scaler.fit_transform(X) # Scale to mean=0, variance=1
# X_test = scaler.fit_transform(X_test)


# Order to be changed
order = 3
poly = PolynomialFeatures(order)
X_poly = poly.fit_transform(X)
print(X_poly)


plt.plot(X, Y, 'o', label = 'trainingsamples')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


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


w_poly = inv(X_poly.T @ X_poly) @ X_poly.T @ Y
print("w_poly is : " + str(w_poly))


w_poly = inv(X_poly.T @ X_poly) @ X_poly.T @ Y
print("w_poly_3 is : " + str(w_poly))

x_figs = np.linspace(0.9, 1.6, 100)
x_figs = x_figs.reshape((100, 1))

P_figs = poly.fit_transform(x_figs)
y_figs_poly = P_figs @ w_poly

plt.plot(x_figs, y_figs_poly, color = 'r', label = "poly regression fit")

plt.plot(X, Y, 'o', label = "Samples")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


y_fit = X_poly @ w_poly
MSE = mean_squared_error(Y, y_fit)
print("MSE is : " + str(MSE))


X_test_poly = poly.fit_transform(X_test)
y_test = X_test_poly @ w_poly
print("y_test is :" + str(y_test))