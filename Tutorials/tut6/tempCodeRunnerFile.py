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