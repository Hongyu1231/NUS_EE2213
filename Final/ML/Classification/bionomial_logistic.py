# ==========================================
#  FILE 3: Binary Classification via Logistic Regression
# ==========================================
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ------------------------------------------
# [用户输入区] USER INPUT AREA
# ------------------------------------------
# 1. 训练数据
X_train = np.array([
    [0.5, 1.2], [-1.0, 0.8], [2.3, -0.7], [0.0, 1.5], [1.1, 0.2], [-0.5, -1.2]
])
y_train = np.array([0, 1, 0, 1, 0, 1])

# 2. 测试数据
X_test = np.array([
    [0.6, 1.1], [-0.8, 0.9]
])
# ------------------------------------------

def run():
    print("--- Binary Classification by Logistic Regression ---")
    
    # 1. 标准化 (逻辑回归对尺度敏感)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. 逻辑回归
    # solver='liblinear' 适合小数据二分类
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_scaled, y_train)
    
    # 3. 预测
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    print("\nModel Parameters:")
    print(f"Weights (w): {model.coef_}")
    print(f"Bias (b): {model.intercept_}")
    
    print("\nTest Predictions:")
    for i, (pred, prob) in enumerate(zip(y_pred, y_proba)):
        # prob[1] 是属于类别 1 的概率 (Sigmoid输出)
        print(f"Sample {i+1}: Prob(Class 1)={prob[1]:.4f} --> Class {pred}")

if __name__ == "__main__":
    run()