import math
from heapq import heappush, heappop

# Graph definition
graph = {'1': {'2': 3, '3': 2},
         '2': {'1': 3, '4': 3, '5': 5},
         '3': {'1': 2, '6': 2, '7': 5},
         '4': {'2': 3, '5': 1},
         '5': {'2': 5, '4': 1, '6': 2},
         '6': {'3': 2, '5': 2, '7': 1},
         '7': {'3': 5, '6': 1}
        }

start = '1'
end = '7'

# Return the shortest path from start node to all nodes in the graph
def dijkstra(graph, start):
    node_data = {node : [math.inf, []] for node in graph}
    visited = set()
    queue = []
    
    # --- ADDED: List to track sequence ---
    popped_sequence = []
    
    node_data[start] = [0, []]
    heappush(queue, (0, start))
    
    while queue:
        cur_dist, node = heappop(queue)
        
        # --- ADDED: Record node ---
        popped_sequence.append(node)
        
        if node in visited:
            continue
        visited.add(node)
        
        for element in graph[node]:
            if element not in visited:
                dist = graph[node][element] + cur_dist
                if dist < node_data[element][0]:
                    node_data[element][0] = dist
                    node_data[element][1] = node_data[node][1] + [node]
                    heappush(queue, (node_data[element][0], element))
                    
    # --- CHANGED: Return both data and sequence ---
    return node_data, popped_sequence


# ==========================================
# Run Function 1 (All Nodes)
# ==========================================
print("--- Dijkstra (All Nodes) ---")
# Unpacking the two return values
result, sequence = dijkstra(graph, start)

print(f"Expansion Order (Popped): {sequence}")
print("-" * 30)

for node in result:
    dist = result[node][0]
    path = result[node][1] + [node]
    if dist == math.inf:
        print(f"There is no path to {node}")
    else:
        print(f"Shortest distance to {node} : {dist}")
        print(f"Shortest path to {node} : {path}")

print("\n" + "="*40 + "\n")


# Specify the start and the goal
def dijkstra_to_goal(graph, start, goal):
    node_data = {node: [math.inf, []] for node in graph}
    visited = set()
    queue = []
    
    # --- ADDED: List to track sequence ---
    popped_sequence = []
    
    node_data[start] = [0, []]
    heappush(queue, (0, start))
    
    while queue:
        cur_dist, node = heappop(queue)
        
        # --- ADDED: Record node ---
        popped_sequence.append(node)
        
        # Check if goal is reached
        if node == goal:
            final_path = node_data[node][1] + [node]
            # --- CHANGED: Return cost, path, AND sequence ---
            return cur_dist, final_path, popped_sequence
        
        if node in visited:
            continue
        visited.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                weight = graph[node][neighbor]
                dist = weight + cur_dist
                
                if dist < node_data[neighbor][0]:
                    node_data[neighbor][0] = dist
                    node_data[neighbor][1] = node_data[node][1] + [node]
                    heappush(queue, (dist, neighbor))
                    
    return None, None, popped_sequence


# ==========================================
# Run Function 2 (Specific Goal)
# ==========================================
print("--- Dijkstra (To Goal) ---")
result = dijkstra_to_goal(graph, start, end)

if result[0] is not None:
    total_cost, path, sequence = result # Unpack 3 values
    print(f"Expansion Order (Popped): {sequence}")
    print(f"Total Cost: {total_cost}")
    print(f"Path: {path}")
else:
    print("No path found.")