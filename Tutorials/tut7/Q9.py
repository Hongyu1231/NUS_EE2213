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

def ReLU(z):
    return np.maximum(0, z)

def Softmax(z, eps=1e-15):
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    pred_Y = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    pred_Y = np.clip(pred_Y, eps, 1 - eps)
    return pred_Y

def cross_entropy_cost(Y, Y_hat):
    return -np.sum(Y * np.log(Y_hat)) / Y.shape[0]

def forward_pass(X, W1, W2, W3):
    O1 = ReLU(X @ W1)

    # second layer output
    # Column of 1s for bias term
    ones = np.ones((O1.shape[0], 1))
    # Concatenate along columns (axis=1)
    A2 = np.hstack((ones, O1))
    O2 = ReLU(A2 @ W2)

    # third layer output
    ones = np.ones((O2.shape[0], 1))
    A3 = np.hstack((ones, O2))
    Y_hat = Softmax(A3 @ W3)

    return Y_hat, A3, A2, O2, O1

# backward pass for output layer
def backward_pass_output(Y, Y_hat, A3, W3, lr):
    N=Y.shape[0]

    # Error at output layer
    E3 = Y_hat - Y
    # Gradient at output layer
    G3 = (A3.T @ E3) / N
    # Weights update at output layer
    W3_new = W3 - lr * G3

    return E3, G3, W3_new

# backward pass for hidden layer
def backward_pass_hidden(E_ladd1, W_ladd1, A_l, O_l, W_l, lr):
    N = A_l.shape[0]
    # Error at hidden layer l
    E_l = E_ladd1 @ W_ladd1[1:].T * (O_l > 0)  # ReLU derivative
    # Gradient at hidden layer l
    G_l = (A_l.T @ E_l)/N
    # Weights update at hidden layer l
    W_l_new = W_l - lr * G_l

    return E_l, G_l, W_l_new

def MLP_3layer(X, Y, W1_init, W2_init, W3_init, lr, num_iters):
    W1, W2, W3 = W1_init.copy(), W2_init.copy(), W3_init.copy() 
    # Create a copy of the initial weights to work out.
    cost_vec = np.zeros(num_iters+1)
    for i in range(1, num_iters+1):
        Y_hat, A3, A2, O2, O1 = forward_pass(X, W1, W2, W3)
        cost = cross_entropy_cost(Y, Y_hat)
        if i % 1000 == 1:
            print(f"Iteration {i-1}, Cost: {cost}")
        cost_vec[i-1]=cost
        E3, G3, W3_new = backward_pass_output(Y, Y_hat, A3, W3, lr) 
        E2, G2, W2_new = backward_pass_hidden(E3, W3, A2, O2, W2, lr)
        E1, G1, W1_new = backward_pass_hidden(E2, W2, X, O1, W1, lr)

        W1, W2, W3 = W1_new.copy(), W2_new.copy(), W3_new.copy()
        # W=W_new.copy(): avoid modifying the original weights. 
        #                 Creates a new array W in memory with the same values as W_new.
        #                 This is a deep copy, so changes to W won’t affect W_new, and vice versa.
        # W=W_new: Only creates a new reference pointing to the same array.
        #          Changes to W would affect W_new, because they share the same memory, and vice versa.

    Y_hat, A3, A2, O2, O1 = forward_pass(X, W1, W2, W3)
    cost = cross_entropy_cost(Y, Y_hat)
    cost_vec[i]=cost
    print(f"Final cost: {cost}")
    
    return Y_hat, cost_vec, W1, W2, W3
    
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

Ytr_est, cost_vec_32, W1_new_32, W2_new_32, W3_new_32 = MLP_3layer(X_train_poly, Ytr_onehot, W1, W2, W3, lr, num_iters)
train_acc = accuracy_score(y_train, np.argmax(Ytr_est, axis=1))
print(f"Training accuracy for learning rate {lr}: {train_acc}")
Yval_est,_,_,_,_= forward_pass(X_val_poly, W1_new_32, W2_new_32, W3_new_32)
val_acc = accuracy_score(y_val, np.argmax(Yval_est, axis=1))
print(f"Validation accuracy for learning rate {lr}: {val_acc}")

# Plot cost function values over iterations for each learning rate
plt.figure(0, figsize=[9,4.5])
plt.rcParams.update({'font.size': 16})
plt.plot(np.arange(0, num_iters+1, 1), cost_vec_32, label=f'hid_size={hid_layer_size}')
plt.legend(loc='upper right', fontsize=15)
plt.xlabel('Iteration Number')
plt.ylabel('Categorical Cross Entropy Cost')
plt.xticks(np.arange(0, num_iters+1, 2000))
plt.title('3-layer MLP')
plt.show()

# evaluate on test set
from sklearn.metrics import confusion_matrix
Yts_est,_,_,_,_= forward_pass(X_test_poly, W1_new_32, W2_new_32, W3_new_32)
test_acc = accuracy_score(y_test, np.argmax(Yts_est, axis=1))
print(f"test accuracy for hidden layer size=32: {test_acc}")
cm_test = confusion_matrix(y_test, np.argmax(Yts_est, axis=1))
print("Confusion Matrix for Test Set:")
print(cm_test)