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

def Depth_First_Search(graph, start):
    queue = [] #A queue for FIFO, aiming to access each node in order
    visited = [] #A list for visited node, we will not access them again
    sequence = [] #The returned value, which is a path that the nodes were visited in order
    queue.append(start) #Initialize the queue
    visited.append(start)
    
    while queue:
        node = queue.pop() #Pop the last element out
        sequence.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
    return sequence
    
print(Depth_First_Search(graph, "A"))