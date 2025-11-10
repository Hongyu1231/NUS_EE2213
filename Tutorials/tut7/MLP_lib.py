import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits() # Load dataset as a dictionary-like object.
print("The digits dataset keys:", digits.keys())
print("The digits dataset description:\n", digits.DESCR)

X, y = digits.data, digits.target
print("The digits dataset features (X) shape:", X.shape)
print("The digits dataset labels (y) shape:", y.shape)

print("flattened vector:", X[0])
print("original image:\n", digits.images[0])
print("label:", y[0])
plt.imshow(digits.images[0], cmap='gray')
plt.title(f'Label: {digits.target[0]}')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Normalize features
scaler = StandardScaler() # Creates a scaler object that standardizes features to have mean = 0 and std = 1.
X_train = scaler.fit_transform(X_train) # Fit and transform the training data
X_val = scaler.transform(X_val) # Transform the validation data
X_test = scaler.transform(X_test) # Transform the test data

# Print the shapes of the datasets
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse_output=False)
reshaped = y_train.reshape(len(y_train), 1) # 1D array to 2D array
Ytr_onehot = onehot_encoder.fit_transform(reshaped)

reshaped = y_val.reshape(len(y_val), 1)
Yval_onehot = onehot_encoder.fit_transform(reshaped)

reshaped = y_test.reshape(len(y_test), 1)
Yts_onehot = onehot_encoder.fit_transform(reshaped)

print(Yts_onehot)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score

order = 1
# Create polynomial features X to P
Poly = PolynomialFeatures(order)
X_train_poly = Poly.fit_transform(X_train)
X_val_poly = Poly.fit_transform(X_val)
X_test_poly = Poly.fit_transform(X_test)

hid_layer_size = 36

# output layer size = number of classes
output_layer_size = Ytr_onehot.shape[1]

# Initialize weights
np.random.seed(42)  # For reproducibility
W1 = np.random.randn(X_train_poly.shape[1], hid_layer_size) # Generates random numbers from a standard normal distribution
W2 = np.random.randn(hid_layer_size+1, hid_layer_size)
W3 = np.random.randn(hid_layer_size+1, output_layer_size)

lr=0.01
num_iters = 20000

from sklearn.neural_network import MLPClassifier

hid_layer_size_list = [32, 36]
train_acc_list = {}
val_acc_list = {}
max_val_acc = 0
best_size = 0

for hid_layer_size in hid_layer_size_list:
    # Define MLP and fit once to initialize shapes
    mlp = MLPClassifier(hidden_layer_sizes=(hid_layer_size, hid_layer_size), 
                        solver='sgd',
                        max_iter=num_iters, 
                        learning_rate_init=lr,
                        random_state=42)
    mlp.fit(X_train, y_train)

    ytr_pred = mlp.predict(X_train)
    train_acc = accuracy_score(y_train, ytr_pred)
    print(f"Training accuracy for hidden layer size {hid_layer_size}: {train_acc}")
    train_acc_list[hid_layer_size]=train_acc

    yval_pred = mlp.predict(X_val)
    val_acc = accuracy_score(y_val, yval_pred)
    print(f"Validation accuracy for hidden layer size {hid_layer_size}: {val_acc}")
    val_acc_list[hid_layer_size]=val_acc

    if val_acc > max_val_acc:
        max_val_acc = val_acc
        best_size = hid_layer_size
        yts_pred=mlp.predict(X_test)
        test_acc = accuracy_score(y_test, yts_pred)

print(f"Best hidden layer size: {best_size}, Max validation accuracy: {max_val_acc}")
print(f"Test accuracy: {test_acc}")
