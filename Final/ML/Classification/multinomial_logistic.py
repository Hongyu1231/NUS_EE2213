# ==========================================
#  FILE 4: Multinomial Classification via Logistic Regression
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
y_train = np.array([0, 1, 2, 0, 2, 1])

# 2. 测试数据
X_test = np.array([
    [0.6, 1.1], [-0.8, 0.9], [2.0, -0.5]
])
# ------------------------------------------

def run():
    print("--- Multinomial Classification by Logistic Regression (Softmax) ---")
    
    # 1. 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. 多分类逻辑回归
    # multi_class='multinomial' 会使用 Softmax Loss (Cross Entropy)
    model = LogisticRegression(solver='lbfgs', random_state=42)
    model.fit(X_scaled, y_train)
    
    # 3. 预测
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    print("\nTest Predictions:")
    for i, (pred, prob) in enumerate(zip(y_pred, y_proba)):
        # prob 包含所有类别的概率 (Softmax输出)
        print(f"Sample {i+1}: Probs={np.round(prob, 4)} --> Class {pred}")

if __name__ == "__main__":
    run()