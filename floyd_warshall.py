
# Reference: https://www.programiz.com/dsa/floyd-warshall-algorithm
from cmath import inf
import numpy as np

# called at the fisrt time when only have a graph
def floyd_warshall(graph, next, index_mapping):
    num_vertices = len(graph.nodes)
    dist = [[np.inf] * num_vertices for _ in range(num_vertices)]
    for i in range(num_vertices):
        dist[i][i] = 0

    # print(f"number of nodes {num_vertices}")
    edges_with_attributes = graph.edges(data=True)
    for edge in edges_with_attributes:
        i, j, attributes = edge # i, j are osmID
        length = attributes.get('length', np.inf)
        index_mapped_i, index_mapped_j = index_mapping[i], index_mapping[j]
        dist[index_mapped_i][index_mapped_j] = length
        if length == np.inf:
            next[index_mapped_i][index_mapped_j] = -1
        else:
            next[index_mapped_i][index_mapped_j] = index_mapped_j
            
    # print(f"number of nodes {num_vertices}")
    # print("Distance Matrix :\n", dist)
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                new_distance = dist[i][k] + dist[k][j]
                if new_distance < dist[i][j]:
                    dist[i][j] = new_distance
                    next[i][j] = next[i][k]
    
    return dist, next    

# used for updating distance matrix caused by blockages
def floyd_warshallblockages(dist, next):
    num_vertices = len(dist)

    #checkInf(dist)

    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                new_distance = dist[i][k] + dist[k][j]
                if new_distance < dist[i][j]:
                    dist[i][j] = new_distance
                    next[i][j] = next[i][k]
    
    return dist, next

def checkInf(dist):
    num_vertices = len(dist)
    for row in range(num_vertices):
        for col in range(num_vertices):
            if dist[row][col] == float('inf'):
                print(f"The element at row {row} and column {col} is infinity.")