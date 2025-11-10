graph = {"A" : ["B", "C"],
         "B" : ["A", "D", "E"],
         "C" : ["A", "F", "I"],
         "D" : ["B", "G"],
         "E" : ["B", "G", "H"],
         "F" : ["C", "H"],
         "G" : ["D", "E", "S"],
         "H" : ["E", "F", "S"],
         "I" : ["C", "J"],
         "J" : ["I", "H", "K"],
         "K" : ["J", "S"],
         "S" : ["G", "H", "K"]}


def Breadth_First_Search(graph,  start, goal):
    visited = [start]
    queue = [start]
    path = {}
    
    while queue:
        node = queue.pop(0)
        
        if node == goal:
            result = []
            while node:
                result.append(node)
                if node != start:
                    node = path[node]
                else:
                    break
            result.reverse()
            return result
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
                path[neighbor] = node
    return None
                
print(Breadth_First_Search(graph,  "A", "S"))