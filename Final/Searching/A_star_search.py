import numpy as np
from heapq import heappush, heappop
import math


# Get heuristic by euclidean distance
def astar_search_eclidien(road_map, city_coordinates, start_city, destination_city):
    """
    Implements A* Search to find the shortest path between two cities.
    Returns: ((shortest_path_list, total_cost), popped_sequence)
    """

    # Helper: Calculate Euclidean distance (Heuristic / H-score)
    def get_heuristic(city):
        coord1 = np.array(city_coordinates[city])
        coord2 = np.array(city_coordinates[destination_city])
        output = (np.linalg.norm(coord1 - coord2))**2
        print(output)
        return output

    # Initialize path tracking and cost tracking
    # g_score: The exact cost from start_city to the current city
    g_scores = {city: math.inf for city in road_map}
    g_scores[start_city] = 0
    
    # path_history: Stores the route taken to reach each city
    path_history = {city: [] for city in road_map}

    visited = set() 
    priority_queue = []
    
    # --- ADDED: List to track the sequence ---
    popped_sequence = []

    # Push tuple: (F-score, City). Note: F = G + H
    start_f_score = 0 + get_heuristic(start_city)
    heappush(priority_queue, (start_f_score, start_city))

    while priority_queue:
        # Pop the city with the lowest F-score
        current_f, current_city = heappop(priority_queue)

        if current_city == destination_city:
            final_path = path_history[current_city] + [current_city]
            return (final_path, g_scores[current_city]), popped_sequence

        # Optimization: If we've already processed this node with a lower cost, skip it
        if current_city in visited:
            continue
        visited.add(current_city)
        popped_sequence.append(current_city)

        # Explore neighbors
        for neighbor, weight in road_map[current_city].items():
            if neighbor in visited:
                continue

            # Calculate the new G-score (cost to get to neighbor via current_city)
            tentative_g = g_scores[current_city] + weight

            if tentative_g < g_scores[neighbor]:
                # Found a better path to neighbor; update data
                g_scores[neighbor] = tentative_g
                path_history[neighbor] = path_history[current_city] + [current_city]
                
                # F-score = New G-score + Heuristic
                f_score = tentative_g + get_heuristic(neighbor)
                heappush(priority_queue, (f_score, neighbor))

    # Return values if no path is found
    return (None, np.inf), popped_sequence

# 1. Define a sample map (Graph)
# Distances are approximate based on the coordinates below
sample_map = {
    'A': {'B': 5, 'C': 10},
    'B': {'A': 5, 'D': 9, 'E': 20},
    'C': {'A': 10, 'D': 3},
    'D': {'B': 9, 'C': 3, 'E': 2}, # D -> E is very short
    'E': {'B': 20, 'D': 2},
    'F': {}
}

# 2. Define city coordinates for the Heuristic (x, y)
# A=(0,0), E is far away at (10, 5)
sample_coords = {
    'A': (0, 0),
    'B': (3, 4),
    'C': (8, 0),
    'D': (8, 3),
    'E': (10, 5),
}

# 3. Define Start and Goal
start = 'A'
goal = 'E'

# 4. Run the Search
(path, cost), sequence = astar_search_eclidien(sample_map, sample_coords, start, goal)

# 5. Print the Results
print("--- Euclidean A* ---")
if path:
    print(f"Expansion Order (Popped): {sequence}")
    print(f"Path found: {path}")
    print(f"Total Cost: {cost}")
else:
    print(f"No path found from {start} to {goal}")
    print(f"Expansion Order (Popped): {sequence}")


print("\n" + "="*30 + "\n")


# Get heuristic by a heuristic dictionary
def astar_search(road_map, heuristic_estimates, start_city, destination_city):
    """
    Implements A* Search using a pre-calculated heuristic dictionary.
    
    :heuristic_estimates: dict
        A dictionary mapping nodes to their estimated cost to the goal.
        Example: {'A': 10, 'B': 8, 'Goal': 0}
        
    Returns: ((shortest_path_list, total_cost), popped_sequence)
    """

    # Helper: Look up the heuristic value
    # We use .get(city, 0) to default to 0 if a city is missing (Dijkstra behavior)
    def get_heuristic(city):
        return heuristic_estimates.get(city, 0)

    # Initialize path tracking and cost tracking
    g_scores = {city: math.inf for city in road_map}
    g_scores[start_city] = 0
    
    path_history = {city: [] for city in road_map}

    visited = set() 
    priority_queue = []
    
    # --- ADDED: List to track the sequence ---
    popped_sequence = []

    # Push tuple: (F-score, City). Note: F = G + H
    start_f_score = 0 + get_heuristic(start_city)
    heappush(priority_queue, (start_f_score, start_city))

    while priority_queue:
        current_f, current_city = heappop(priority_queue)
        
        # --- ADDED: Record the node ---
        popped_sequence.append(current_city)

        if current_city == destination_city:
            final_path = path_history[current_city] + [current_city]
            return (final_path, g_scores[current_city]), popped_sequence

        if current_city in visited:
            continue
        visited.add(current_city)

        for neighbor, weight in road_map[current_city].items():
            if neighbor in visited:
                continue

            tentative_g = g_scores[current_city] + weight

            if tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                path_history[neighbor] = path_history[current_city] + [current_city]
                
                f_score = tentative_g + get_heuristic(neighbor)
                heappush(priority_queue, (f_score, neighbor))

    return (None, math.inf), popped_sequence

# 1. Define a sample map (Graph)
sample_map = {
    'S': {'A': 2, 'B': 6},
    'A': {'B': 1, 'S': 2},
    'B': {'C': 1, 'A': 1, 'S': 6, 'D': 2},
    'C': {'G': 1, 'B': 1},
    'D': {'B': 2, 'G': 1},
    'G': {'C': 1, 'D': 1}
}

h_table = {
    'S': 7,
    'A': 14,
    'B': 9, 
    'C': 3,
    'D': 4,
    'G': 0
}

# 3. Run the Search
start_node = 'S'
goal_node = 'G'

(path, cost), sequence = astar_search(sample_map, h_table, start_node, goal_node)

# 4. Print the Results
print("--- Dictionary A* ---")
if path:
    print(f"Expansion Order (Popped): {sequence}")
    print(f"Path found: {path}")
    print(f"Total Cost: {cost}")
else:
    print(f"No path found from {start_node} to {goal_node}")
    print(f"Expansion Order (Popped): {sequence}")