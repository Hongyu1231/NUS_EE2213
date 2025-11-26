import numpy as np

# Given data (To be changed)
w = 0.5  # initial w
eta = 2  # learning rate

# Function to be changed
f = lambda x : x**2/2

# Update w
for i in range(10):
    # Compute the gradient
    # Gradient to be changed
    grad = w
    w = w - eta * grad
    print("Grad: ", grad)
    print(f"Updated w for {i + 1} iterations:", w)

print("\nGradient (∇C(w)):", grad)
print("\nUpdated w:", w)
print("\nFinal Output is:", f(w))

