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

visited = []


def Depth_First_Search_Re(graph, node, visited):
    visited.append(node)
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            Depth_First_Search_Re(graph, neighbor, visited)
            
Depth_First_Search_Re(graph, "A", visited)
print(visited)