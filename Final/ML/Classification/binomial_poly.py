import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

# ==========================================
#  [用户输入区] USER INPUT AREA
# ==========================================
# 1. 训练数据 (二分类标签建议: 0, 1)
# 脚本会自动将 0 映射为 -1，以便使用 sign() 函数
X_train = np.array([
    [0.5, 1.2], [-1.0, 0.8], [2.3, -0.7], [0.0, 1.5], [1.1, 0.2], [-0.5, -1.2]
])
y_train = np.array([0, 1, 0, 1, 0, 1])

# 2. 测试数据
X_test = np.array([
    [0.6, 1.1], [-0.8, 0.9]
])

# 3. 多项式阶数
POLY_DEGREE = 2
# ==========================================

def run():
    print(f"--- Binary Classification via Poly Regression (using sign()) ---")
    
    # [关键步骤] 标签映射: 0 -> -1, 1 -> 1
    # 这样回归平面的零点就是决策边界
    y_train_mapped = np.where(y_train == 0, -1, 1)
    print(f"Mapped Labels: {y_train_mapped}")

    # 1. 特征工程
    poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
    X_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # 2. 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    X_test_scaled = scaler.transform(X_test_poly)
    
    # 3. 线性回归拟合
    model = LinearRegression()
    model.fit(X_scaled, y_train_mapped)
    
    # 4. 预测
    y_pred_raw = model.predict(X_test_scaled)
    
    # 5. 决策: 使用 sign() 函数
    # > 0 -> Class 1
    # < 0 -> Class -1 (即原来的 0)
    y_pred_sign = np.sign(y_pred_raw)
    
    # 映射回 0/1 以便阅读
    y_pred_class = np.where(y_pred_sign == -1, 0, 1)
    
    print("\nTest Predictions:")
    for i, (raw, cls) in enumerate(zip(y_pred_raw, y_pred_class)):
        print(f"Sample {i+1}: Raw={raw:.4f} | Sign={int(np.sign(raw))} -> Class {cls}")

if __name__ == "__main__":
    run()