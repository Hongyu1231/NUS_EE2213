from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, matrix_rank, det
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[1.6], [1.5], [1.0], [1.1], [1.2], [0.9]])
Y = np.array([3.1, 3.4, 2.9, 3.0, 3.8, 2.9])

plt.plot(X, Y, 'o', label = 'trainingsamples')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

order = 1
linear = PolynomialFeatures(order)
X_linear = linear.fit_transform(X)
print(X_linear)

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

def check_inverse_det(matrix):
    deter = det(matrix)
    print("Determinant is : " + str(deter))
    if deter != 0:
        print("Matrix is invertible.")
    else:
        print("Matrix isn't invertible.")

check_inverse_rank(X_linear.T @ X_linear)
check_inverse_det(X_linear.T @ X_linear)

w_linear = inv(X_linear.T @ X_linear) @ X_linear.T @ Y
print("w_linear is : " + str(w_linear))

y_fit = X_linear @ w_linear
plt.plot(X, y_fit, color = 'r', label = "linear regression fit")

plt.plot(X, Y, 'o', label = "Samples")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

MSE = mean_squared_error(Y, y_fit)
print("MSE is : " + str(MSE))

X_test = np.array([[0.5]])
X_test_linear = linear.fit_transform(X_test)
y_test = X_test_linear @ w_linear
print("y_test is :" + str(y_test))