import heapq

# 1. Same Graph and Heuristics
graph = {
    'S': {'A': 2, 'B': 6},
    'A': {'B': 1, 'S': 2},
    'B': {'C': 1, 'A': 1, 'S': 6, 'D': 2},
    'C': {'G': 1, 'B': 1},
    'D': {'B': 2, 'G': 1},
    'G': {'C': 1, 'D': 1}
}

heuristics = {
    'S': 7,
    'A': 14,
    'B': 9, 
    'C': 3,
    'D': 4,
    'G': 0
}

start = 'S'
end = 'G'

def greedy_best_first_search(graph, heuristics, start, goal):
    frontier = []
    # Priority Queue stores: (heuristic, node_name)
    heapq.heappush(frontier, (heuristics[start], start))
    
    came_from = {start: None}
    popped_sequence = [] # List to store the expansion order
    
    print(f"{'Node':<10} | {'H-Value (Priority)':<20}")
    print("-" * 35)

    while frontier:
        # Pop the node with the lowest H value
        current_h, current_node = heapq.heappop(frontier)
        
        # Add to our sequence list
        popped_sequence.append(current_node)
        
        # Print current status
        print(f"{current_node:<10} | {current_h:<20}")

        if current_node == goal:
            print("-" * 35)
            return reconstruct_path(came_from, graph, start, goal), popped_sequence

        for neighbor, weight in graph[current_node].items():
            if neighbor not in came_from:
                heapq.heappush(frontier, (heuristics[neighbor], neighbor))
                came_from[neighbor] = current_node
    
    return (None, 0), popped_sequence

def reconstruct_path(came_from, graph, start, goal):
    current = goal
    path = []
    total_cost = 0
    while current != start:
        path.append(current)
        parent = came_from[current]
        total_cost += graph[parent][current]
        current = parent
    path.append(start)
    path.reverse()
    return path, total_cost

# --- Run the Code ---
(result, cost), sequence = greedy_best_first_search(graph, heuristics, start, end)

if result:
    print(f"\nExpansion Order (Popped): {sequence}")
    print(f"Final Path Found:       {' -> '.join(result)}")
    print(f"Total Path Cost:        {cost}")
else:
    print("No path found.")