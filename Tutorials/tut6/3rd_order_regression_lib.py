import numpy as np
from sklearn.preprocessing import PolynomialFeatures
# Linear regression using libraries
from sklearn.linear_model import LinearRegression
# 3rd Polynormial regression using libraries
X = np.array([[1, 1.5], [0.9, 2], [3, 1]])
y = np.array([1.0, 2.6, 3.0])
order = 3
Poly_3 = PolynomialFeatures(order)
X_Poly_3 = Poly_3.fit_transform(X)

model = LinearRegression(fit_intercept=False)
# fit_intercept=False: when you already augmented 1.
model.fit(X_Poly_3, y)
print("Model coefficients: " + str(model.coef_))
print("Model intercept: " + str(model.intercept_))

# prediction
X_test = np.array([[0.5]])
X_test_poly_3 = Poly_3.fit_transform(X_test)
print("y_test is : " + str(model.predict(X_test_poly_3)))