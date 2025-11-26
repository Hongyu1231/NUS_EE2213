# Import libraries
from pulp import *  # PuLP: Python library for modeling linear programming problems

# Define the LP problem
prob = LpProblem("Maximize_Profit", LpMaximize)

# Decision variables
x1 = LpVariable("x1", lowBound=0)   
x2 = LpVariable("x2", lowBound=0)   

# Objective function (Change these)
prob += 40 * x1 + 30 * x2        # Total profit

# Constraints (Change these)
prob += 2 * x1 + 1 * x2 <= 100   # Machine time
prob += x2 >= 10                  # Minimum product B
prob += x1 + x2 <= 40            # Production capacity

# Solve the LP problem
prob.solve() 

# Results
x1_opt = value(x1) # Get the optimal value of decision variable x1
x2_opt = value(x2) # Get the optimal value of decision variable x2
optimum_value = value(prob.objective) # Get the maximum profit from the objective function

# Output results
print("Optimal number of units to produce:")
print("A:", x1_opt)
print("B:", x2_opt)
print("Optimum Value: $", optimum_value)

