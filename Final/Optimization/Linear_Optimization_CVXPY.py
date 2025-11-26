# Import libraries
import cvxpy as cp  # For solving linear programs

# 1. Define variables (Change these)
x1 = cp.Variable(name="x1", integer=True)    # integer variable
x2 = cp.Variable(name="x2", integer=True) 
x3 = cp.Variable(name="x3", integer=True)    # integer variable
x4 = cp.Variable(name="x4", integer=True)
x5 = cp.Variable(name="x4", integer=True)
# y1 = cp.Variable(name="x1", integer=True)    # integer variable
# y2 = cp.Variable(name="x2", integer=True) 
# y3 = cp.Variable(name="x3", integer=True)    # integer variable
# y4 = cp.Variable(name="x4", integer=True)
# z1 = cp.Variable(name="x1", integer=True)    # integer variable
# z2 = cp.Variable(name="x2", integer=True) 
# z3 = cp.Variable(name="x3", integer=True)    # integer variable
# z4 = cp.Variable(name="x4", integer=True)


# 2. Define objective
objective = cp.Maximize(40*x1 + 75*x2 + 110*x3 + 140*x4 + 90*x5)

# 3. Define constraints (Change these)
constraints = [
    8*x1 + 18*x2 + 25*x3 + 35*x4 + 20*x5 <= 600,
    4*x1 + 7*x2 + 10*x3 + 13*x4 + 8*x5 <= 250,
    x1 + 3*x2 + 5*x3 + 6*x4 + 4*x5 <= 120,
    x1 >= 10,
    x2 <= 15,
    x3 >= 12,
    x4 >= 12,
    x5 >= x1/2,
    x1 >= 0,
    x2 >= 0,
    x3 >= 0,
    x4 >= 0,
    x5 >= 0
]

# 4. Creates the CVXPY optimization problem by combining the objective and the constraints
prob = cp.Problem(objective, constraints)

# 5. Solve the LP problem
prob.solve() 

# 6. Results
x1_opt = x1.value # Get the optimal value of decision variable x1
x2_opt = x2.value # Get the optimal value of decision variable x2
x3_opt = x3.value # Get the optimal value of decision variable x3
x4_opt = x4.value # Get the optimal value of decision variable x4
x5_opt = x5.value # Get the optimal value of decision variable x5
# y1_opt = y1.value # Get the optimal value of decision variable x1
# y2_opt = y2.value # Get the optimal value of decision variable x2
# y3_opt = y3.value # Get the optimal value of decision variable x3
# y4_opt = y4.value # Get the optimal value of decision variable x4
# z1_opt = z1.value # Get the optimal value of decision variable x1
# z2_opt = z2.value # Get the optimal value of decision variable x2
# z3_opt = z3.value # Get the optimal value of decision variable x3
# z4_opt = z4.value # Get the optimal value of decision variable x4
optimum_value = prob.value # Get the maximum profit from the objective function
problem_status = prob.status # Get the solution status

# Output results
print("\nSolution Status:", problem_status)
print("Optimal number of units to produce:")
print("x1:", x1_opt)
print("x2:", x2_opt)
print("x3:", x3_opt)
print("x4:", x4_opt)
print("x5:", x5_opt)
# print("y1:", y1_opt)
# print("y2:", y2_opt)
# print("y3:", y3_opt)
# print("y4:", y4_opt)
# print("z1:", z1_opt)
# print("z2:", z2_opt)
# print("z3:", z3_opt)
# print("z4:", z4_opt)
print("Optimum Value: $", optimum_value)
