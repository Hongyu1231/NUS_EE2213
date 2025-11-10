import sympy as sp

grid_size = 3

P = {}
W = {}
B = {}
S = {}

for x in range(1, grid_size+1):
    for y in range(1, grid_size+1):
        P[(x, y)] = sp.symbols(f'P_{x}_{y}')
        W[(x, y)] = sp.symbols(f'W_{x}_{y}')
        B[(x, y)] = sp.symbols(f'B_{x}_{y}')
        S[(x, y)] = sp.symbols(f'S_{x}_{y}')

def get_neighbors(x, y, grid_size):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = [(x + dx, y + dy) for dx, dy in directions]
    return [(nx, ny) for nx, ny in neighbors if 1 <= nx <= grid_size and 1 <= ny <= grid_size]

def Pit_implies_Breeze(x, y, P, B, grid_size):
    adjacent_Breeze = sp.And(*[B[adj] for adj in get_neighbors(x, y, grid_size)])
    pit_implies_breeze = sp.Implies(P[(x, y)], adjacent_Breeze)
    return pit_implies_breeze

def Breeze_for_Pit(x, y, B, P, grid_size):
    adjacent_pit = sp.Or(*[P[adj] for adj in get_neighbors(x, y, grid_size)])
    breeze_for_pit = sp.Equivalent(B[(x, y)], adjacent_pit)
    return breeze_for_pit

def Wumpus_implies_Stench(x, y, W, S, grid_size):
    adjacent_stench = sp.And(*[S[adj] for adj in get_neighbors(x, y, grid_size)])
    wumpus_implies_stench = sp.Implies(W[(x,y)], adjacent_stench)
    return wumpus_implies_stench

def Stench_for_Wumpus(x, y, S, W, grid_size):
    adjacent_wumpus = sp.Or(*[W[adj] for adj in get_neighbors(x, y, grid_size)])
    stench_for_wumpus = sp.Or(*[W[adj] for adj in get_neighbors(x, y, grid_size)])
    stench_for_wumpus = sp.Equivalent(S[(x, y)], adjacent_wumpus)
    return stench_for_wumpus

def at_least_one_pit(P, grid_size):
    return sp.Or(*[P[(x, y)] for x in range(1, grid_size+1) for y in range(1, grid_size+1)])

def at_least_one_wumpus(W, grid_size):
    return sp.Or(*[W[(x, y)] for x in range(1, grid_size+1) for y in range(1, grid_size+1)])

rules = set()

# Step 3: Combine all rules into a knowledge base
rules = set()

for x in range(1, grid_size+1):
    for y in range(1, grid_size+1):
        pit_implies_breeze = Pit_implies_Breeze(x,y,P,B,grid_size)
        rules.add(pit_implies_breeze)
        breeze_for_pit = Breeze_for_Pit(x,y,B,P,grid_size)
        rules.add(breeze_for_pit)
        wumpus_implies_stench = Wumpus_implies_Stench(x,y,W,S,grid_size)
        rules.add(wumpus_implies_stench)     
        stench_for_wumpus = Stench_for_Wumpus(x,y,S,W,grid_size)
        rules.add(stench_for_wumpus)
rules.add(at_least_one_pit(P, grid_size))
rules.add(at_least_one_wumpus(W, grid_size))

KB = rules.copy()
current_loc = (1, 1)

KB.add(sp.Not(B[current_loc]))
KB.add(sp.Not(S[current_loc]))
KB.add(sp.Not(P[current_loc]))
KB.add(sp.Not(W[current_loc]))

KB_AND = sp.And(*KB)
query = W[(2, 2)]
entails = not sp.satisfiable(sp.And(KB_AND, sp.Not(query)))
entails_not = not sp.satisfiable(sp.And(KB_AND, query))

query = P[(2, 2)]

def Infer(KB_AND, query):
    entails = not sp.satisfiable(sp.And(KB_AND, sp.Not(query)))
    if entails:
        print(query, "is definitely true.")
    else:
        entails_not = not sp.satisfiable(sp.And(KB_AND, query))
        if entails_not:
            print(query, "is definitely false.")
        else:
            print(query, "is uncertain.")

Infer(KB_AND, query)

neighbors = get_neighbors(current_loc[0], current_loc[1], grid_size)

for neighbor in neighbors:
    Infer(KB_AND, W[neighbor])
    Infer(KB_AND, P[neighbor])

KB.add(sp.Not(P[(2,1)]))
KB.add(sp.Not(W[(2,1)]))
KB.add(sp.Not(P[(1,2)]))
KB.add(sp.Not(W[(1,2)]))

current_loc = (2, 1)

KB.add(B[current_loc])
KB.add(sp.Not(S[current_loc]))

KB_AND = sp.And(*KB)

neighbors = get_neighbors(current_loc[0], current_loc[1], grid_size)
for neighbor in neighbors:
    Infer(KB_AND, W[neighbor])
    Infer(KB_AND, P[neighbor])