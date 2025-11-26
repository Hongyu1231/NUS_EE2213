import numpy as np
from sklearn.preprocessing import PolynomialFeatures
# 6rd polynormial regression with ridge using libraries
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

# Training data to be changed
X = np.array([[45, 26, 3], [68, 19, 7], [32, 25, 5], [55, 35, 9], [28, 21, 2]]) # 3 features with 4 samples (The number of numbers in the inner paranthesis is the number of features)
y = np.array([2, 5, 3, 7, 1])

# Test data to be changed
X_test = np.array([[30, 18, 4]])

# Order to be changed
order = 2
poly = PolynomialFeatures(order)

# lambda to be changed
lamda = 1

model_ridge = make_pipeline(poly,Ridge(alpha=lamda, fit_intercept=False))  # poly already includes the bias term,
# therefore fit_intercept=False for Ridge to match the manual implementation.

model_ridge.fit(X, y)
ridge = model_ridge.named_steps['ridge']
# named_steps: a dictionary of all steps in the pipeline.
# model_ridge.named_steps['ridge'] returns the actual Ridge regressor object inside the pipeline
print("Model coefficients: " + str(ridge.coef_))
print("Model intercept: " + str(ridge.intercept_))

# prediction
print("y_test is : " + str(model_ridge.predict(X_test)))