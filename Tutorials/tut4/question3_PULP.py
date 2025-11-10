import pulp

prob = pulp.LpProblem("Simple_problem", pulp.LpMaximize)


x = pulp.LpVariable("x", lowBound = 0)
y = pulp.LpVariable("y", lowBound = 0)

prob += 40*x +30*y

prob += 2*x + y <= 100
prob += y >=10
prob +=x + y <= 40

prob.solve()
print("x = ", x.value(), "y = ", y.value())