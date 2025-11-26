# Graph to be changed
graph = { '1': {'2', '3'},
          '2': {'6'},
          '3': {'4', '5'},
          '4': {},
          '5': {},
          '6': {}
         }


# Perform a Breadth-First Search (BFS) traversal starting from node "start"
def Breadth_First_Search(graph, start):
    queue = [] #A queue for FIFO, aiming to access each node in order
    visited = [] #A list for visited node, we will not access them again
    sequence = [] #The returned value, which is a path that the nodes were visited in order
    queue.append(start) #Initialize the queue
    visited.append(start)
    
    while queue:
        node = queue.pop(0) #Pop the first element out
        sequence.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
    return sequence
        
print(Breadth_First_Search(graph, "1"))
# Uncomment and modify the upper line if want to traverse all the graph with BFS



# Perform a Depth-First Search (DFS) traversal starting from node
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
    
#print(Depth_First_Search(graph, "A"))
# Uncomment and modify the upper line if want to traverse all the graph with DFS



# Perform a Depth-First Search (DFS) with recursion traversal starting from node
visited = []
def Depth_First_Search_Re(graph, node, visited):
    visited.append(node)
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            Depth_First_Search_Re(graph, neighbor, visited)
            
Depth_First_Search_Re(graph, "1", visited)
print(visited)
# Uncomment and modify the upper line if want to traverse all the graph with DFS(Recursion)
