import cvxpy as cp

x_A = cp.Variable(name = "product_A", integer = True)
x_B = cp.Variable(name = "product_B", integer = True)

objective = cp.Maximize(40 * x_A + 30 * x_B)

constraint = [
    x_A >= 0,
    x_B >= 0,
    2 * x_A +x_B <= 100,
    x_A + x_B <= 40,
    x_B >= 10
]

prob = cp.Problem(objective, constraint)

prob.solve()

print("Solver Status: ", prob.status)

print("Product_A: ", x_A.value)
print("Product_B: ", x_B.value)