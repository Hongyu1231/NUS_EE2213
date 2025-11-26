import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

# ==========================================
#  [用户输入区] USER INPUT AREA
# ==========================================
# 1. 训练数据 (标签: 0, 1, 2...)
X_train = np.array([
    [0.5, 1.2], [-1.0, 0.8], [2.3, -0.7], [0.0, 1.5], [1.1, 0.2], [-0.5, -1.2]
])
y_train = np.array([0, 1, 2, 0, 2, 1])

# 2. 测试数据
X_test = np.array([
    [0.6, 1.1], [-0.8, 0.9], [2.0, -0.5]
])

# 3. 多项式阶数
POLY_DEGREE = 2
# ==========================================

def run():
    print(f"--- Multinomial Classification via Poly Regression (using argmax) ---")
    
    # 1. 标签 One-Hot 编码
    # 例如: 0 -> [1, 0, 0], 1 -> [0, 1, 0]
    enc = OneHotEncoder(sparse_output=False)
    y_onehot = enc.fit_transform(y_train.reshape(-1, 1))
    
    # 2. 特征工程
    poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
    X_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # 3. 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    X_test_scaled = scaler.transform(X_test_poly)
    
    # 4. 多输出回归拟合
    model = LinearRegression()
    model.fit(X_scaled, y_onehot)
    
    # 5. 预测 (输出的是分数矩阵，Shape: [N_samples, N_classes])
    y_scores = model.predict(X_test_scaled)
    
    # 6. 决策: Argmax
    y_pred_class = np.argmax(y_scores, axis=1)
    
    print("\nTest Predictions:")
    for i, (scores, cls) in enumerate(zip(y_scores, y_pred_class)):
        print(f"Sample {i+1}: Scores={np.round(scores, 3)} -> Max Index (Class): {cls}")

if __name__ == "__main__":
    run()