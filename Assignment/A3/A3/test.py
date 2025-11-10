import numpy as np
from A3_A0311920U import optimize_shipments, gradient_descent


# ----------------------------
# Problem 1 - Stress & Edge
# ----------------------------
def test_problem1_stress_scale():
    """
    把供需整体放大 N 倍（结构相同），验证可行并满足约束。
    由于是线性成本+整数变量，最优成本会按比例放大（但我们只验证约束与整数性）。
    """
    base_supply = np.array([50, 30, 40])
    base_demand = np.array([20, 45, 25, 30])
    C = [
        [10, 25, 30, 20],
        [12, 32, 25, 22],
        [35, 20, 15, 40],
    ]
    for N in [2, 5, 10, 50, 100]:
        supply = (base_supply * N).tolist()
        demand = (base_demand * N).tolist()
        cost, X = optimize_shipments(supply, demand, C)
        assert np.all(X.sum(axis=1) <= np.array(supply)), f"Supply exceeded for scale {N}"
        assert np.all(X.sum(axis=0) == np.array(demand)), f"Demand mismatch for scale {N}"
        assert np.issubdtype(X.dtype, np.integer), f"Not integer for scale {N}"
    print("✅ Problem 1 stress_scale passed.")


def test_problem1_stress_random_many():
    """
    多次随机测试：随机供需（总量匹配）、随机成本，确保解满足约束。
    注意：次数太多会慢；这里取较合理次数。
    """
    rng = np.random.default_rng(123)
    trials = 20
    for t in range(trials):
        supply = rng.integers(5, 200, size=3).tolist()
        total_supply = sum(supply)

        demand_raw = rng.integers(1, 150, size=4)
        demand = (demand_raw / demand_raw.sum() * total_supply).astype(int).tolist()
        demand[-1] += total_supply - sum(demand)  # 调整最后一个，确保总量相等

        cost_matrix = rng.integers(1, 100, size=(3, 4)).tolist()

        cost, X = optimize_shipments(supply, demand, cost_matrix)
        assert np.all(X.sum(axis=1) <= np.array(supply)), f"[t={t}] Supply exceeded"
        assert np.all(X.sum(axis=0) == np.array(demand)), f"[t={t}] Demand mismatch"
        assert np.issubdtype(X.dtype, np.integer), f"[t={t}] Not integer dtype"
    print("✅ Problem 1 stress_random_many passed.")


def test_problem1_infeasible():
    """
    不可行性测试：供给总量 < 需求总量，期望优化器失败或抛异常。
    你的实现若抛 RuntimeError 就算通过；若返回值也应让测试失败。
    """
    supply = [10, 0, 0]          # total 10
    demand = [8, 8, 0, 0]        # total 16 -> infeasible
    C = [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    failed = False
    try:
        optimize_shipments(supply, demand, C)
    except Exception:
        failed = True
    assert failed, "Infeasible instance should raise or fail to solve."
    print("✅ Problem 1 infeasible case passed.")


# ----------------------------
# Problem 2 - Stress & Edge
# ----------------------------
def test_problem2_long_iters():
    """
    超长迭代测试：num_iters=100000，验证：
    1) w_out 接近 5
    2) f_out 接近 1
    3) f_out 单调不增（允许浮点微小数值噪声）
    """
    lr = 0.1
    iters = 100000
    w_out, f_out = gradient_descent(learning_rate=lr, num_iters=iters)
    assert abs(w_out[-1] - 5.0) < 1e-6, "w did not approach 5 sufficiently"
    assert abs(f_out[-1] - 1.0) < 1e-9, "f(w) did not approach 1 sufficiently"
    # 单调不增：允许极小浮点误差
    assert np.all(f_out[1:] <= f_out[:-1] + 1e-12), "f_out is not non-increasing"
    print("✅ Problem 2 long_iters passed.")


def test_problem2_lr_near_boundary():
    """
    学习率接近上界（<0.2）时仍应稳定收敛。
    使用 0.199 与 0.1999 测试，并验证 f_out 单调下降、最终接近 1。
    """
    for lr in [0.199, 0.1999]:
        w_out, f_out = gradient_descent(learning_rate=lr, num_iters=500)
        assert f_out[-1] <= f_out[0], f"f did not decrease for lr={lr}"
        assert f_out[-1] >= 1.0 - 1e-9, "f should not go below 1"
        assert np.all(f_out[1:] <= f_out[:-1] + 1e-10), f"f not non-increasing for lr={lr}"
    print("✅ Problem 2 lr_near_boundary passed.")


def test_problem2_small_lr_monotone():
    """
    很小的学习率应当严格单调下降，但收敛慢。
    """
    w_out, f_out = gradient_descent(learning_rate=1e-4, num_iters=2000)
    assert f_out[-1] < f_out[0], "f did not decrease for small lr"
    assert np.all(f_out[1:] <= f_out[:-1] + 1e-12), "f not non-increasing for small lr"
    print("✅ Problem 2 small_lr_monotone passed.")


if __name__ == "__main__":
    # Problem 1
    test_problem1_stress_scale()
    test_problem1_stress_random_many()
    test_problem1_infeasible()

    # Problem 2
    test_problem2_long_iters()
    test_problem2_lr_near_boundary()
    test_problem2_small_lr_monotone()

    print("🎉 All stress & edge tests passed.")
