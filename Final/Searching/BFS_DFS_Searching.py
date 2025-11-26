# Graph to be changed
graph = { "A" : ["B", "C" ], 
          "B" : ["A", "D", "E"], 
          "C" : ["A", "F"], 
          "D" : ["B", "G"],
          "E" : ["B", "G"],
          "F" : ["C", "H"],
          "G" : ["D", "E", "S"],
          "H" : ["E", "F", "S"],
          "S" : ["G", "H"]
         }

# ==========================================
# 1. Breadth-First Search (BFS)
# ==========================================
def Breadth_First_Search(graph,  start, goal):
    visited = [start]
    queue = [start]
    path = {}
    
    # --- ADDED: List to track pop sequence ---
    pop_sequence = []
    
    while queue:
        node = queue.pop(0)
        
        # --- ADDED: Record the node ---
        pop_sequence.append(node)
        
        if node == goal:
            result = []
            while node:
                result.append(node)
                if node != start:
                    node = path[node]
                else:
                    break
            result.reverse()
            # --- CHANGED: Return path AND sequence ---
            return result, pop_sequence
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
                path[neighbor] = node
    return None, pop_sequence

# Test BFS
print("--- BFS ---")
bfs_path, bfs_seq = Breadth_First_Search(graph, "A", "S")
print(f"Path: {bfs_path}")
print(f"Pop Sequence: {bfs_seq}")
print("\n")


# ==========================================
# 2. Depth-First Search (Iterative)
# ==========================================
def Depth_First_Search(graph, start, goal):
    visited = [start]
    stack = [start] 
    path = {}
    
    # --- ADDED: List to track pop sequence ---
    pop_sequence = []
    
    while stack:
        node = stack.pop()
        
        # --- ADDED: Record the node ---
        pop_sequence.append(node)
        
        if node == goal:
            result = []
            while node:
                result.append(node)
                if node != start:
                    node = path[node]
                else:
                    break
            result.reverse()
            # --- CHANGED: Return path AND sequence ---
            return result, pop_sequence
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.append(neighbor)
                stack.append(neighbor)
                path[neighbor] = node
    return None, pop_sequence

# Test DFS (Iterative)
print("--- DFS (Iterative) ---")
dfs_path, dfs_seq = Depth_First_Search(graph, "A", "S")
print(f"Path: {dfs_path}")
print(f"Pop Sequence: {dfs_seq}")
print("\n")


# ==========================================
# 3. Depth-First Search (Recursive)
# ==========================================
# I added 'pop_sequence' as an argument to track the order across recursions
def Depth_First_Search_Re(graph, node, goal, visited, pop_sequence):
    visited.append(node)
    
    # --- ADDED: Record the node (Process order) ---
    pop_sequence.append(node)
    
    if node == goal:
        return [node]
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            path = Depth_First_Search_Re(graph, neighbor, goal, visited, pop_sequence)
            
            if path is not None:
                return [node] + path
                
    return None

# Test DFS (Recursive)
print("--- DFS (Recursive) ---")
visited_list = []
seq_list = [] # List to hold the sequence
dfs_re_path = Depth_First_Search_Re(graph, "A", "S", visited_list, seq_list)
print(f"Path: {dfs_re_path}")
print(f"Pop Sequence: {seq_list}")