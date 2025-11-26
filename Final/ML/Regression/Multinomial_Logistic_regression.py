import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder

# Data to be changed
X = np.array([[1.2, -0.4, 0.8], [-0.6, 2.0, -0.5], [0.3, -1.2, 1.7], [2.1, 0.5, -0.8]])
y_raw = np.array([[3], [1], [2], [3]])
lr = 0.1
iteration = 1 # 我稍微增加了迭代次数，让结果更明显一点，你可以改回 1

# 1. 定义你想要预测的新数据 (例如 2 个样本)
X_test = np.array([
    [1.0, -0.5, 0.5],   # Sample 1
    [-0.5, 1.5, -0.2]   # Sample 2
])

# Step 1: initialize weights
W = np.array([[0, 0, 0], [0.02, -0.01, 0.03], [-0.05, 0.04, 0.01], [0.03, 0.02, -0.02]])

def multi_logistic_cost_gradient(X, W, Y, eps=1e-15):
    # Compute prediction, cost and gradient based on cross entropy
    z = X @ W
    # 为了数值稳定性，通常建议减去最大值，但对于简单演示可以直接 exp
    exp_z = np.exp(z) 
    pred_Y = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    
    # 计算 Cost (加上 eps 防止 log(0))
    cost = np.sum(-(Y * np.log(pred_Y + eps))) / X.shape[0]
    gradient = X.T @ (pred_Y - Y) / X.shape[0]

    return pred_Y, cost, gradient


# Step 0: prepare data
poly = PolynomialFeatures(1)
P = poly.fit_transform(X) # fit_transform 用于训练数据

onehot_encoder = OneHotEncoder(sparse_output=False)
Y = onehot_encoder.fit_transform(y_raw)

# Step 2: perform gradient descent
print("Training...")
for i in range(iteration):
    pred_Y, cost, gradient = multi_logistic_cost_gradient(P, W, Y, eps=1e-15)
    W = W - lr * gradient

print(f"Updated weights after {iteration} iteration: \n", W)
print("-" * 30)


# ==========================================
# Step 3: Prediction on Test Data (新增部分)
# ==========================================

print("Testing...")

# 2. 预处理: 使用之前的 poly 对象 transform (注意是用 transform, 不是 fit_transform)
# 这会自动加上第一列的 bias (1.0)
P_test = poly.transform(X_test)

# 3. 计算 Logits (Z)
z_test = P_test @ W

# 4. 计算概率 (Softmax)
exp_z_test = np.exp(z_test)
probs_test = exp_z_test / np.sum(exp_z_test, axis=-1, keepdims=True)

# 5. 获取预测类别的索引 (Argmax)
# 这会返回 0, 1, 2 这样的索引
pred_indices = np.argmax(probs_test, axis=1)

# 6. 将索引映射回原始标签 (Original Labels)
# onehot_encoder.categories_[0] 里面存的是 [1, 2, 3]
# 如果 index 是 0，对应的就是 Label 1
predicted_labels = onehot_encoder.categories_[0][pred_indices]

print("Test Input:\n", X_test)
print("Predicted Probabilities:\n", probs_test)
print("Predicted Labels:", predicted_labels)