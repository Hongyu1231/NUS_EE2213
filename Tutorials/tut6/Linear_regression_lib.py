import numpy as np
# Linear regression using libraries
from sklearn.linear_model import LinearRegression

X = np.array([[3,15,50], [5,12,68], [2,9,60],[4,19,80]])
y = np.array([5,6,3,8])
model = LinearRegression() # Automatic intercept: By default, LinearRegression adds the intercept
model.fit(X, y)
print("Model coefficients: " + str(model.coef_))
print("Model intercept: " + str(model.intercept_))

# prediction
X_test = np.array([[7,10,70]])
print("y_test is : " + str(model.predict(X_test)))
