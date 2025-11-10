from sympy import symbols, And, Not, Or, Implies, Equivalent
from sympy.logic.boolalg import truth_table

# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A4_A0311920U(query):
    """
    Args:
        query: A sympy logical expression representing the query to be checked
               (e.g., A, Not(B), etc.).

    Returns:
        result: A string "True" if the query is a Knight;
                A string "False" if the query is a Knave;
                A string "Not Sure" if the type of the query cannot be determined.
    """

    # your code goes here
    A, B, C = symbols('A, B, C')
    all_symbols = [A, B, C]
    KB_Alex = Equivalent(A, Not(B))
    KB_Ben = Equivalent(B, Not(Equivalent(A, B)))
    KB_Chloe = Equivalent(C, Or(A, C))
    KB = And(KB_Alex, KB_Ben, KB_Chloe)

    def check_entailment(kb, alpha, symbols_list):
        all_models = truth_table(kb, symbols_list)

        for row in all_models:
            minterm = row[0]
            kb_is_true = row[-1]
            
            if kb_is_true:
                model_dict = dict(zip(symbols_list, minterm))
                alpha_is_true = alpha.subs(model_dict)
                
                if not alpha_is_true:
                    return False
        return True

    entails_query = check_entailment(KB, query, all_symbols)
    entails_not_query = check_entailment(KB, Not(query), all_symbols)
    if entails_query:
        result = "True"
    elif entails_not_query:
        result = "False"
    else:
        result = "Not Sure"
        
    # return in this order
    return result


