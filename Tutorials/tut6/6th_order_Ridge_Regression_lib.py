import numpy as np
from sklearn.preprocessing import PolynomialFeatures
# 6rd polynormial regression with ridge using libraries
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

X = np.array([[1.6],[1.5], [1.0], [1.1], [1.2], [0.9]])
y = np.array([3.1, 3.4, 2.9, 3.0, 3.8, 2.9])
order = 6
Poly_6 = PolynomialFeatures(order)

lamda = 0.0001

model_ridge = make_pipeline(Poly_6,Ridge(alpha=lamda, fit_intercept=False))  # Poly_6 already includes the bias term,
# therefore fit_intercept=False for Ridge to match the manual implementation.

model_ridge.fit(X, y)
ridge = model_ridge.named_steps['ridge']
# named_steps: a dictionary of all steps in the pipeline.
# model_ridge.named_steps['ridge'] returns the actual Ridge regressor object inside the pipeline
print("Model coefficients: " + str(ridge.coef_))
print("Model intercept: " + str(ridge.intercept_))

# prediction
X_test = np.array([[0.5]])
print("y_test is : " + str(model_ridge.predict(X_test)))