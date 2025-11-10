from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, matrix_rank, det
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[1.6], [1.5], [1.0], [1.1], [1.2], [0.9]])
Y = np.array([3.1, 3.4, 2.9, 3.0, 3.8, 2.9])
X_test = np.array([[0.5]])
order = 6
Poly_6 = PolynomialFeatures(order)
X_Poly_6 = Poly_6.fit_transform(X)

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

check_inverse_rank(X_Poly_6.T @ X_Poly_6)
check_inverse_det(X_Poly_6.T @ X_Poly_6)

x_figs_poly_6 = np.linspace(0.9, 1.6, 100)
x_figs_poly_6 = x_figs_poly_6.reshape((-1, 1))
P_figs_6 = Poly_6.fit_transform(x_figs_poly_6)

X_test_poly6 = Poly_6.fit_transform(X_test)
print(X_Poly_6)

lamda_list = [0.0001, 0.01, 1]
color_list = ["g", "r", "c"]

plt.figure(0, figsize = [9, 4.5])
plt.rcParams.update({'font.size' : 16})
plt.plot(X, Y, 'o', color = 'k', label = 'traininf=g samples')

for color, lamda in zip(color_list, lamda_list):
    reg_L = lamda * np.identity(X_Poly_6.shape[1])
    w_poly_6 = inv(X_Poly_6.T @ X_Poly_6 + reg_L) @ X_Poly_6.T @ Y
    print(f"w_poly_6 with λ={lamda} is : " + str(w_poly_6))

    y_pred_poly_6 = X_Poly_6 @ w_poly_6
    MSE = mean_squared_error(Y, y_pred_poly_6)
    print(f"MSE with λ={lamda} is : " + str(MSE))

    y_test_poly_6 = X_test_poly6 @ w_poly_6
    print(f"y_test_poly_6 with λ={lamda} is : " + str(y_test_poly_6))
    print("\n")

    y_figs_Poly6 = P_figs_6 @ w_poly_6
    plt.plot(x_figs_poly_6, y_figs_Poly6, color = color, label = f'6th order poly fit (λ={lamda})')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

