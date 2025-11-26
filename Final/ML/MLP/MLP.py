import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ==========================================
#  >>> 1. 用户配置 (PROMPT AREA) <<<
# ==========================================
MLP_CONFIG = {
    "layer_structure": [64, 32, 32, 10], 
    "hidden_activation": "relu", 
    "output_activation": "softmax",
    "loss_function": "cross_entropy",
    "learning_rate": 0.01,
    "iterations": 20000
}

# ==========================================
#  >>> 2. 核心类定义 <<<
# ==========================================
class FinalMLP:
    def __init__(self, config):
        self.structure = config["layer_structure"]
        self.hidden_act = config["hidden_activation"]
        self.output_act = config["output_activation"]
        self.loss_type = config["loss_function"]
        self.lr = config["learning_rate"]
        self.iterations = config["iterations"]
        
        self.weights = []
        self.cost_history = []
        
        np.random.seed(42) 
        
        # 初始化权重
        for i in range(len(self.structure) - 1):
            input_dim = self.structure[i]
            output_dim = self.structure[i+1]
            # 包含 Bias 行，所以行数是 input_dim + 1
            W = np.random.randn(input_dim + 1, output_dim)
            self.weights.append(W)

    def _add_bias(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack((ones, X))

    def _activation(self, z, name, derivative=False):
        if name == "relu":
            if derivative: return (z > 0).astype(float)
            return np.maximum(0, z)
        elif name == "sigmoid":
            s = 1 / (1 + np.exp(-z))
            if derivative: return s * (1 - s)
            return s
        elif name == "softmax":
            exps = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exps / np.sum(exps, axis=1, keepdims=True)
        elif name == "identity":
            if derivative: return np.ones_like(z)
            return z
        return z

    def _compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        if self.loss_type == "cross_entropy":
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.sum(y_true * np.log(y_pred)) / m
        elif self.loss_type == "square_loss":
            return np.mean(np.square(y_true - y_pred))
        return 0

    def forward(self, X):
        self.caches = [] 
        self.z_caches = []
        
        A = self._add_bias(X) 
        
        # Hidden Layers
        for i in range(len(self.weights) - 1):
            W = self.weights[i]
            self.caches.append(A)
            Z = np.dot(A, W)
            self.z_caches.append(Z)
            A_out = self._activation(Z, self.hidden_act)
            A = self._add_bias(A_out)

        # Output Layer
        W_last = self.weights[-1]
        self.caches.append(A)
        Z_last = np.dot(A, W_last)
        self.z_caches.append(Z_last)
        A_final = self._activation(Z_last, self.output_act)
        
        return A_final

    def backward(self, y_true, y_pred):
        m = y_true.shape[0]
        grads = []
        num_layers = len(self.weights)
        
        # Output Layer Gradient
        if self.loss_type == "cross_entropy" and self.output_act == "softmax":
            dZ = y_pred - y_true
        elif self.loss_type == "square_loss" and self.output_act == "identity":
            dZ = 2 * (y_pred - y_true)
        else:
            dZ = y_pred - y_true 

        A_in_last = self.caches[-1]
        dW_last = np.dot(A_in_last.T, dZ) / m
        grads.insert(0, dW_last)
        
        # Hidden Layers Gradients
        for i in reversed(range(num_layers - 1)):
            W_next = self.weights[i + 1]
            W_next_no_bias = W_next[1:, :] 
            
            dA = np.dot(dZ, W_next_no_bias.T)
            Z_curr = self.z_caches[i]
            dZ = dA * self._activation(Z_curr, self.hidden_act, derivative=True)
            
            A_in_curr = self.caches[i]
            dW = np.dot(A_in_curr.T, dZ) / m
            grads.insert(0, dW)

        # Update Weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads[i]

    def train(self, X, y):
        print(f"Status: Start training... (Total iterations: {self.iterations})")
        for i in range(self.iterations):
            y_pred = self.forward(X)
            cost = self._compute_loss(y, y_pred)
            self.cost_history.append(cost)
            self.backward(y, y_pred)
            
            # 每1000次打印一次
            if i % 1000 == 0:
                print(f"Iteration {i:5d} | Cost: {cost:.5f}")
        
        print(f"Status: Training Finished! Final Cost: {self.cost_history[-1]:.5f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

# ==========================================
#  >>> 3. 立即执行区域 (EXECUTION) <<<
# ==========================================

# 1. 准备数据
print("Status: Loading data...")
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

enc = OneHotEncoder(sparse_output=False)
y_train_onehot = enc.fit_transform(y_train.reshape(-1, 1))

# 2. 实例化并训练
print("Status: Model Initialized.")
model = FinalMLP(MLP_CONFIG)
model.train(X_train, y_train_onehot)

# 3. 打印权重 (Output W)
print("\n" + "="*60)
print(f"{'FINAL WEIGHTS (W) INSPECTION':^60}")
print("="*60)

for i, W in enumerate(model.weights):
    print(f"\n>>> Layer {i+1} Weights <<<")
    print(f"Shape: {W.shape} (Rows = PrevLayer + 1 Bias, Cols = NextLayer)")
    print("-" * 30)
    
    # 打印部分权重以防刷屏
    if W.shape[0] > 8:
        print(W[:4, :]) 
        print(f"   ... (omitting {W.shape[0]-8} rows) ...")
        print(W[-4:, :])
    else:
        print(W)

# 4. 验证准确率
from sklearn.metrics import accuracy_score
test_pred = model.predict(X_test)
acc = accuracy_score(y_test, test_pred)
print("\n" + "="*60)
print(f"TEST ACCURACY: {acc:.4f}")
print("="*60)

# 5. 画图
plt.plot(model.cost_history)
plt.title("Cost Curve")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid(True)
plt.show()