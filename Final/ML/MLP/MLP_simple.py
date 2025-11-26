import numpy as np
from sklearn.preprocessing import OneHotEncoder

# ==========================================
#  1. 核心数学库 (Math Kernel)
# ==========================================

def get_activation(name):
    """
    [用法指南] 根据层的位置选择激活函数:
    1. 'relu':    最推荐用于【隐藏层】。它解决梯度消失问题，计算速度快。
    2. 'softmax': 必须用于【多分类输出层】。它将数值转化为概率分布 (和为1)。
    3. 'sigmoid': 用于【二分类输出层】。它将数值压缩到 (0, 1) 之间。
    4. 'linear':  用于【回归任务输出层】。不做任何处理，直接输出数值。
    """
    if name == 'relu': return lambda z: np.maximum(0, z)
    if name == 'sigmoid': return lambda z: 1 / (1 + np.exp(-z))
    if name == 'linear': return lambda z: z
    
    if name == 'softmax': 
        # Softmax 公式: exp(z) / sum(exp(z))
        # 技巧: 减去最大值 (z - max) 是为了防止 exp(z) 溢出，不影响结果，但更稳定
        return lambda z: np.exp(z - np.max(z, axis=1, keepdims=True)) / np.sum(np.exp(z - np.max(z, axis=1, keepdims=True)), axis=1, keepdims=True)
    
    raise ValueError(f"Unknown activation: {name}")

def get_derivative(name):
    """获取激活函数的导数 (用于反向传播计算梯度)"""
    if name == 'relu': return lambda o: (o > 0).astype(float)
    if name == 'linear': return lambda o: np.ones_like(o)
    if name == 'sigmoid': return lambda o: o * (1 - o)
    if name == 'softmax': return lambda o: np.ones_like(o) # 占位符，Softmax导数在 Delta 中特殊处理
    raise ValueError(f"Unknown derivative: {name}")

def calculate_loss_value(y, y_hat, loss_type):
    """
    [用法指南] 损失函数必须与输出层激活函数匹配:
    1. 'cross_entropy'        <--> 搭配 'softmax' (多分类任务)
    2. 'binary_cross_entropy' <--> 搭配 'sigmoid' (二分类任务)
    3. 'mse' (均方误差)       <--> 搭配 'linear'  (回归/预测数值任务)
    """
    epsilon = 1e-15 # 防止 log(0) 报错
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    N = y.shape[0]
    
    if loss_type == 'mse': 
        return np.mean(0.5 * (y_hat - y)**2)
    elif loss_type == 'binary_cross_entropy': 
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    elif loss_type == 'cross_entropy': 
        return -np.sum(y * np.log(y_hat)) / N
    raise ValueError(f"Unknown loss: {loss_type}")

def compute_output_delta(y, y_hat, activation_type, loss_type):
    """
    计算输出层的误差项 (Delta)。
    这里包含了一个数学上的“魔法简化”：
    
    当使用 [Softmax + CrossEntropy] 或 [Sigmoid + BCE] 时，
    复杂的求导公式最后会神奇地抵消，简化为： (预测值 - 真实值)
    
    这就是为什么深度学习总是成对使用它们的原因。
    """
    # --- 黄金组合 (Golden Combinations) ---
    # 梯度简化为: y_hat - y
    if (loss_type == 'cross_entropy' and activation_type == 'softmax') or \
       (loss_type == 'binary_cross_entropy' and activation_type == 'sigmoid') or \
       (loss_type == 'mse' and activation_type == 'linear'):
        return y_hat - y
    
    # --- 非标准组合 (需要乘导数) ---
    # 例如: 在回归任务中强行用 Relu 输出
    if loss_type == 'mse':
        return (y_hat - y) * get_derivative(activation_type)(y_hat)
    
    raise ValueError(f"Unsupported combination: {loss_type} + {activation_type}")

# ==========================================
#  2. 核心逻辑 (逻辑未变，增加注释)
# ==========================================

def forward_pass_flexible(X, weights_list, activations_list, use_bias_list):
    caches = {} 
    A_curr = X
    caches['A_input'] = [] 
    caches['O_output'] = []
    
    for i, W in enumerate(weights_list):
        use_bias = use_bias_list[i]
        
        # [Bias 处理]
        # 如果 use_bias=True，我们在输入矩阵左侧拼一个全为 1 的列
        # X 形状变化: (N, 3) -> (N, 4)
        if use_bias:
            ones = np.ones((A_curr.shape[0], 1))
            A_curr_with_bias = np.hstack((ones, A_curr))
        else:
            A_curr_with_bias = A_curr
            
        # [维度检查] 确保矩阵乘法 A @ W 能进行
        if A_curr_with_bias.shape[1] != W.shape[0]:
            raise ValueError(f"[Layer {i+1} Error] Input dim {A_curr_with_bias.shape[1]} != Weight row {W.shape[0]}.")

        caches['A_input'].append(A_curr_with_bias)

        # [核心计算] Z = A @ W
        Z = A_curr_with_bias @ W
        
        # [激活] O = f(Z)
        act_func = get_activation(activations_list[i])
        O = act_func(Z)
        caches['O_output'].append(O)
        
        A_curr = O 
            
    return O, caches

def backward_pass_flexible(y, y_hat, caches, weights_list, activations_list, lr, loss_type, use_bias_list):
    N = y.shape[0]
    new_weights_list = [w.copy() for w in weights_list]
    
    # 1. 计算最后一层的误差
    E_curr = compute_output_delta(y, y_hat, activations_list[-1], loss_type)
    
    # 2. 从后往前传播
    for i in reversed(range(len(weights_list))):
        A_prev = caches['A_input'][i]
        W_curr = weights_list[i]
        use_bias = use_bias_list[i]
        
        # [梯度下降] W = W - lr * Gradient
        # Gradient = Input.T @ Error / N
        G_curr = (A_prev.T @ E_curr) / N
        new_weights_list[i] = W_curr - lr * G_curr
        
        # [传递误差给前一层]
        if i > 0:
            O_prev = caches['O_output'][i-1]
            d_act_func_prev = get_derivative(activations_list[i-1])
            d_sigma = d_act_func_prev(O_prev)
            
            # 如果使用了 Bias，第一行权重是连在常数 1 上的，不回传误差
            # 所以我们只取 W[1:, :]
            if use_bias:
                W_backprop = W_curr[1:, :]
            else:
                W_backprop = W_curr
            
            E_prev = (E_curr @ W_backprop.T) * d_sigma
            E_curr = E_prev 

    return new_weights_list

# ==========================================
#  3. 通用实验室 (General Lab)
# ==========================================

if __name__ == "__main__":
    np.set_printoptions(precision=12, suppress=True, linewidth=200)

    print("🧪 神经网络实验室: Softmax + CrossEntropy 示例")

    # --- A. 数据 (Data) ---
    X = np.array([
        [0.277, -0.017, -0.111],
        [0.255, -0.024, -0.097],
        [0.279, -0.019, -0.110],
        [0.220, -0.035, -0.113],
        [0.217, -0.032, -0.112]
    ]) 
    y_raw = np.array([[1], [1], [2], [3], [3]]) 
    
    # [One-Hot 编码]
    # Softmax 输出的是概率 (如 [0.1, 0.8, 0.1])
    # 我们的标签也必须变成概率格式 (如 [0, 1, 0]) 才能计算 loss
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y_raw)

    # --- B. 常用配置组合 (Configuration) ---
    
    # 场景 1: 多分类 (当前配置)
    # 激活: 最后一层用 'softmax'
    # Loss: 用 'cross_entropy'
    activations = ['relu', 'relu', 'softmax']
    loss_type = 'cross_entropy'
    
    # 场景 2: 二分类 (如需测试，请取消注释)
    # activations = ['relu', 'sigmoid']
    # loss_type = 'binary_cross_entropy'
    
    # 场景 3: 回归预测 (如需测试，请取消注释)
    # activations = ['relu', 'linear']
    # loss_type = 'mse'

    learning_rate = 0.1
    num_epochs = 2
    
    # [Bias 开关]
    # 只有当你知道自己在做什么时才设为 False
    # False 意味着该层必须经过原点 (0,0)，通常会降低模型能力
    #记得修改 W 里的 bias 行
    use_bias_list = [True, True, True]

    # --- C. 权重 (Weights) ---
    # Layer 1 (Bias=True, 所以行数=4)
    W1 = np.array([
        [0, 0.1, -0.05],
        [0.2, -0.1, 0.4],
        [-0.3, 0.25, 0.1],
        [0.05, -0.2, 0.3]   # Input 3
    ])
    
    # Layer 2 (Bias=True, 所以行数=4)
    W2 = np.array([
        [0, 0.1, -0.05],
        [0.2, -0.1, 0.4],
        [-0.3, 0.25, 0.1],
        [0.05, -0.2, 0.3]   # Input 3
    ])

    W3 = np.array([
        [0, 0.1, -0.05],
        [0.2, -0.1, 0.4],
        [-0.3, 0.25, 0.1],
        [0.05, -0.2, 0.3]   # Input 3
    ])
    
    weights = [W1, W2, W3]

    # --- D. 运行 (Run) ---
    for epoch in range(num_epochs):
        # 1. Forward
        y_hat, caches = forward_pass_flexible(X, weights, activations, use_bias_list)
        
        # 2. Loss
        loss = calculate_loss_value(y, y_hat, loss_type)
        
        # 3. Backward
        weights = backward_pass_flexible(y, y_hat, caches, weights, activations, learning_rate, loss_type, use_bias_list)
        
        print(f"\n>>> Epoch {epoch+1} Result <<<")
        print(f"Loss ({loss_type}): {loss:.12f}")
        
        print("\n--- Prediction (Probabilities) ---")
        # 加上注释说明输出什么
        print("每一行是一个样本，每一列是该类别的概率:")
        print(y_hat)
        
        print("\n--- Updated Weights ---")
        for i, W in enumerate(weights):
            print(f"Layer {i+1}:")
            print(W)

    print("\n✅ Done.")