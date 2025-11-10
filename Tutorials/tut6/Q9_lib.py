import numpy as np
from numpy.linalg import inv,matrix_rank,det
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris

# Load the California housing dataset
data = load_iris() # Load as a DataFrame

X = data.data
y = data.target
print(X)
print(y)

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Print the shapes of the datasets
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print("model coefficients:", model.coef_)
print("model intercept:", model.intercept_)
y_pred_train = model.predict(X_train)
train_acc = accuracy_score(y_train, y_pred_train)
y_pred_val = model.predict(X_val)
val_acc = accuracy_score(y_val, y_pred_val)
y_pred_test = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)
print(f"Training accuracy using sklearn: {train_acc}")
print(f"Validation accuracy using sklearn: {val_acc}")
print(f"Test accuracy using sklearn: {test_acc}")