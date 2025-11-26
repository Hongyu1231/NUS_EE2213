import numpy as np
from sklearn.preprocessing import PolynomialFeatures
# Linear regression using libraries
from sklearn.linear_model import LinearRegression


# Training Data to be changed (No need to augmented 1)
X = np.array([[45, 26, 3], [68, 19, 7], [32, 25, 5], [55, 35, 9], [28, 21, 2]]) # 3 features with 4 samples (The number of numbers in the inner paranthesis is the number of features)
y = np.array([2, 5, 3, 7, 1])

# Test data to be changed
X_test = np.array([[30, 18, 4]])

order = 2 # Order to be changed
Poly = PolynomialFeatures(order)
X_Poly = Poly.fit_transform(X)


model = LinearRegression()
model.fit(X_Poly, y)
# The coefficient: [1, x1, x2, x3, x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2, .......]
print("Model coefficients: " + str(model.coef_))
print("Model intercept: " + str(model.intercept_))


# prediction
# X_test_poly = Poly.transform(X_test)
# print("y_test is : " + str(model.predict(X_test_poly)))