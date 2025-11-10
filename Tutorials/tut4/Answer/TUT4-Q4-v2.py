# Import libraries
from pulp import *

# Define the problem
prob = LpProblem("T4Q4", LpMinimize)

# Define decision variables
x_1 = LpVariable("x1", lowBound=0, cat='Continuous') 
x_2 = LpVariable("x2", lowBound=0, cat='Continuous') 

# Objective function
prob += 2 * x_1 + x_2       

# Constraints
prob += x_1 + x_2 >= 5  
prob += x_1 + 2 * x_2 >= 6   

# Solve the problem
prob.solve()

# Results
x1_opt = value(x_1) # Get the optimal value of decision variable x_1
x2_opt = value(x_2) # Get the optimal value of decision variable x_2
min_value = value(prob.objective) # Get the maximum profit from the objective function


# Output results

# Check and display the result status
status_code = prob.status              # Get numeric status code
status_text = LpStatus[status_code]    # Convert to readable text

print("Solver Status:", status_text)   # Show what happened


if status_text == "Optimal":
        print("Optimal Solution Found!!")
        print("x1:", x1_opt)
        print("x2:", x2_opt)
        print("Optimal objective value:", min_value)
elif status_text == "Unbounded":
        print("Unbounded solution -- need more constraints")
elif status_text == "Infeasible":
        print("Problem has no feasible solution")
else: 
        print("Problem has an undefined status -- something went wrong.")

#print(LpStatus)