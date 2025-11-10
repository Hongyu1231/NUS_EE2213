from sympy import symbols, And, Not, Or, Implies, Equivalent

A = symbols('A')
B = symbols('B')
C = symbols('C')
D = symbols('D')

statement_A = Implies(A, B)
statement_B = Implies(C, Not(A))
statement_C = And(Or(B, C), Not(And(B, C)))

KB = set()
KB.add(statement_A)
KB.add(statement_B)
KB.add(statement_C)

# from sympy.logic.boolalg import truth_table
# for assignment, Truth_value in truth_table(KB, [A, B, C, D]):
#     print(f"[{A} = {'Ture' if assignment[0] else 'False'}, {B} = {'Ture' if assignment[1] else 'False'}] KB = {'True' if Truth_value else 'False'}")

from sympy.logic.inference import satisfiable

def contradiction_check(KB, query):
    return not satisfiable(And(KB, Not(query)))
print("KB entails A?", contradiction_check(KB, A))
print("KB entails B?", contradiction_check(KB, B))
print("KB entails C?", contradiction_check(KB, C))
# print("KB entails (not B)?", contradiction_check(KB, Not(B)))