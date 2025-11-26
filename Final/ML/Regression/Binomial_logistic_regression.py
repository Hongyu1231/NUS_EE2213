import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# 1. Data
X = np.array([[25, 15, 0],[4, 3, 3],[40, 30, 1],[2, 2, 5], [15, 10, 1]])
# We use the raw binary labels (0 or 1) directly. No OneHotEncoder needed.
y_raw = np.array([[1], [0], [1], [0], [1]]) 

lr = 0.5
iteration = 20
order = 1

X_test = np.array([[1, 10, 5, 5]])

# 2. Prepare Data (Polynomial Features)
poly = PolynomialFeatures(order)
P = poly.fit_transform(X) # Shape: (4, 4)

# 3. Initialize Weights
# Standard Binomial uses ONE column of weights (Shape: 4x1)
W = np.array([[0.5], [-0.1], [0.6], [1.5]])

# --- Helper Functions ---

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_logistic_cost_gradient(X, W, Y, eps=1e-15):
    # 1. Prediction (Sigmoid)
    z = X @ W
    pred_Y = sigmoid(z)  # Probability that y=1
    
    # 2. Cost (Binary Cross Entropy / Log Loss)
    # Formula: -mean( y*log(p) + (1-y)*log(1-p) )
    cost = -np.mean(Y * np.log(pred_Y + eps) + (1 - Y) * np.log(1 - pred_Y + eps))
    
    # 3. Gradient
    # The derivative simplifies to the exact same structure as Softmax!
    # X.T * (Prediction - Actual) / N
    gradient = X.T @ (pred_Y - Y) / X.shape[0]

    return pred_Y, cost, gradient

# 4. Gradient Descent Step
for i in range(iteration):
    pred_Y, cost, gradient = binary_logistic_cost_gradient(P, W, y_raw)
    W = W - lr * gradient

print(f"Updated weights after {iteration} iteration: \n", W)
print(f"Predictions (Probability of Class 1): \n", pred_Y)

pred = sigmoid(X_test @ W)
print(pred)