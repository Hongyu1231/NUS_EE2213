from sklearn.metrics import mean_squared_error
import numpy as np
from numpy.linalg import inv
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# --- 1. VARIABLES TO CHANGE ---

# Training Data: You can now add as many columns (features) as you want
# Example: 6 samples, 2 features
X = np.array([[45, 26, 3], [68, 19, 7], [32, 25, 5], [55, 35, 9], [28, 21, 2]]) # 3 features with 4 samples (The number of numbers in the inner paranthesis is the number of features)
Y = np.array([2, 5, 3, 7, 1])

# Test data to be changed
X_test = np.array([[30, 18, 4]])

# Polynomial Order to be changed
order = 2

# Ridge Regression Lambdas (Regularization strength)
lamda_list = [1]

# ------------------------------

# 2. SCALING (Crucial for Multi-Feature Stability)
# We fit the scaler on Training data, and apply it to both Train and Test
# scaler = StandardScaler()
# X_ = scaler.fit_transform(X)
# X_test = scaler.transform(X_test) # Apply same scale to test

# 3. POLYNOMIAL TRANSFORM
poly = PolynomialFeatures(order)
X_poly = poly.fit_transform(X)
X_test_poly = poly.transform(X_test) # Apply same poly rules to test

print(f"Input Features: {X.shape[1]}")
print(f"Polynomial Features Generated: {X_poly.shape[1]}")
print("-" * 30)

# 4. RIDGE REGRESSION LOOP
for lamda in lamda_list:
    # Create Regularization Matrix (Identity * Lambda)
    # We use the size of the polynomial matrix (X_poly)
    reg_L = lamda * np.identity(X_poly.shape[1])

    # Calculate Weights: w = (X^T X + λI)^-1 X^T Y
    # Note: X_poly.T @ X_poly works for ANY number of features automatically
    Matrix = X_poly.T @ X_poly + reg_L
    
    # We use inv() here because Ridge (λ>0) guarantees the matrix is invertible
    w_poly = inv(Matrix) @ X_poly.T @ Y
    
    # Training Error
    y_pred = X_poly @ w_poly
    MSE = mean_squared_error(Y, y_pred)

    # Prediction on Test Data
    y_test_pred = X_test_poly @ w_poly

    print(f"Lambda: {lamda}")
    print(f"  MSE: {MSE:.5f}")
    print(f"  Prediction: {y_test_pred}")
    print("-" * 30)