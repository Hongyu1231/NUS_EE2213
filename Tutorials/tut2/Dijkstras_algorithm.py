import math
from heapq import heappush, heappop

graph = {"a" : {'b' : 15, 'c' : 20, 'd' : 10},
         'b' : {'d' : 10, 'e' : 15},
         'c' : {'b' : 5, 'e' : 10, 'g' : 30},
         'd' : {'e' : 5, 'f' : 10},
         'e' : {'g' : 5},
         'f' : {'g' : 10},
         'g' : {}
        }

def dijkstra(graph, start):
    node_data = {node : [math.inf, []] for node in graph}
    visited = set()
    queue = []
    
    node_data[start] = [0, []]
    heappush(queue, (0, start))
    
    while queue:
        cur_dist, node = heappop(queue)
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
    return node_data



result = dijkstra(graph, 'a')
for node in result:
    dist = result[node][0]
    path = result[node][1] + [node]
    if dist == math.inf:
        print(f"There is no path to {node}")
    print(f"Shortest distance to {node} : {dist}")
    print(f"Shortest path to {node} : {path}")