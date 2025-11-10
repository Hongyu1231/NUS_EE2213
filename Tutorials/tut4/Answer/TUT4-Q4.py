# Import libraries
import cvxpy as cp  # For solving linear programs

# 1. Define variables
x1 = cp.Variable()  # defines a continuous decision variable x_1.
x2 = cp.Variable()
x3 = cp.Variable()
x4 = cp.Variable()
x5 = cp.Variable()# defines a continuous decision variable x_2.

# 2. Define objective
objective = cp.Maximize(2*x1 - 3*x2 +x3)

# 3. Define constraints
constraints = [
    x1 - x2 + x3 <= 5,
    x1 - x2 + 4*x3 <= 7,
    x1 + 2*x2 -x3 + x4 <= 14,
    x3 -x4 +x5 <= 7,
    x1>=-15,
    x2>=-15,
    x3>=-15,
    x4>=-15,
    x5>=-15,
    x1<=15,
    x2<=15,
    x3<=15,
    x4<=15,
    x5<=15,
    
    
    
]

# 4. Create a CVXPY problem object
prob = cp.Problem(objective, constraints)

# 5. Solve the LP problem
prob.solve() 

# 6. Results
x1_opt = x1.value  # Get the optimal value of decision variable x_1
x2_opt = x2.value  # Get the optimal value of decision variable x_2
min_obj = prob.value  # Get the minimum from the objective function
problem_status = prob.status  # Get the solution status

# Output results
print("\nSolution Status:", problem_status)
print("x1:", x1_opt)
print("x2:", x2_opt)
print("Minimum objective value:", min_obj)
