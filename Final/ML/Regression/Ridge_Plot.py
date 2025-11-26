from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


# Input data to be changed (Must keep 1 feature to be plotted)
X = np.array([[1.6], [1.5], [1.0], [1.1], [1.2], [0.9]])
Y = np.array([3.1, 3.4, 2.9, 3.0, 3.8, 2.9])

# Test data to be changed (Must keep 1 feature to be plotted)
X_test = np.array([[0.5]])

# Order to be changed
order = 6
Poly = PolynomialFeatures(order)
X_Poly = Poly.fit_transform(X)

# Testing lambda value to be changed
lamda_list = [0.0001, 0.01, 1, 3, 5]

# To be changed according to the number of lambda_list
color_list = ["g", "r", "c", 'b', "#853E75"]


X_figs_poly = np.linspace(0.9, 1.6, 100)
X_figs_poly = X_figs_poly.reshape((-1, 1))
P_figs = Poly.fit_transform(X_figs_poly)


X_test_poly = Poly.fit_transform(X_test)
print(X_Poly)


plt.figure(0, figsize = [9, 4.5])
plt.rcParams.update({'font.size' : 16})
plt.plot(X, Y, 'o', color = 'k', label = 'traininf=g samples')

for color, lamda in zip(color_list, lamda_list):
    reg_L = lamda * np.identity(X_Poly.shape[1])
    w_poly = inv(X_Poly.T @ X_Poly + reg_L) @ X_Poly.T @ Y
    print(f"w_poly with λ={lamda} is : " + str(w_poly))

    y_pred_poly = X_Poly @ w_poly
    MSE = mean_squared_error(Y, y_pred_poly)
    print(f"MSE with λ={lamda} is : " + str(MSE))

    y_test_poly = X_test_poly @ w_poly
    print(f"y_test_poly with λ={lamda} is : " + str(y_test_poly))
    print("\n")

    y_figs_poly = P_figs @ w_poly
    plt.plot(X_figs_poly, y_figs_poly, color = color, label = f'Polynomial order poly fit (λ={lamda})')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

