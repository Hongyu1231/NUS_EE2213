from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, matrix_rank, det
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[1, 1.5], [0.9, 2], [3, 1]])
Y = np.array([1.0, 2.6, 3.0])
X_test = np.array([[0.5]])
order = 3
Poly_3 = PolynomialFeatures(order)
X_Poly_3 = Poly_3.fit_transform(X)
print(X_Poly_3)

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

check_inverse_rank(X_Poly_3.T @ X_Poly_3)
check_inverse_det(X_Poly_3.T @ X_Poly_3)

w_Poly_3 = inv(X_Poly_3.T @ X_Poly_3) @ X_Poly_3.T @ Y
print("w_poly_3 is : " + str(w_Poly_3))

x_figs = np.linspace(0.9, 1.6, 100)
x_figs = x_figs.reshape((100, 1))

P_figs_3 = Poly_3.fit_transform(x_figs)
y_figs_poly_3 = P_figs_3 @ w_Poly_3

plt.plot(x_figs, y_figs_poly_3, 'r', label = '3rd order poly regression fit')

plt.plot(X, Y, 'o', label = 'samples')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

y_pred_poly_3 = X_Poly_3 @ w_Poly_3
MSE = mean_squared_error(Y, y_pred_poly_3)
print("MSE is : " + str(MSE))

X_test_poly_3 = Poly_3.fit_transform(X_test)
y_test_poly_3 = X_test_poly_3 @ w_Poly_3
print("y_test_poly_3 is : " + str(y_test_poly_3))



