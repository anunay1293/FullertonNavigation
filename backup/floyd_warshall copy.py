"""
GROUP 1: Anunay Amrit, Angelica Cabato, Pranav Vijay Chand, Riya Chapatwala, Sai Satya Jagannadh Doddipatla, Nhat Ho

Dr. Shah

CPSC 535: Advanced Algorithms (Spring 2024)

"""
# Reference: https://www.programiz.com/dsa/floyd-warshall-algorithm
import numpy as np

def floyd_warshall(graph, index_mapping):
    num_vertices = len(graph.nodes)
    # initialize graph
    dist = np.full((num_vertices, num_vertices), np.inf)
    np.fill_diagonal(dist, 0)

    # Initialize predecessor matrix
    pred = np.empty((num_vertices, num_vertices), dtype=np.int64)
    pred.fill(-1)  # Use -1 to indicate no predecessor

    # print(f"number of nodes {num_vertices}")
    edges_with_attributes = graph.edges(data=True)
    for edge in edges_with_attributes:
        i, j, attributes = edge # i, j are original index
        
        # return
        length = attributes.get('length', np.inf)
        index_mapped_i, index_mapped_j = index_mapping[i], index_mapping[j]
        dist[index_mapped_i, index_mapped_j] = length
        pred[index_mapped_i, index_mapped_j] = index_mapped_i
        print(f"index_mapped_i {index_mapped_i} index_mapped_j {index_mapped_j} : {length}")

    print(f"number of nodes {num_vertices}")
    print("Distance Matrix before:\n", dist)
    # return
    # iterate over the graph and update the matrix if new shortest path is found
    for k in range(num_vertices):
        print(f"-----------------------kkkkkkkkkkkkkkkk------------------------: {k}")
        for i in range(num_vertices):
            for j in range(num_vertices):
                new_distance = dist[i, k] + dist[k, j]
                print(f"new_distance {new_distance}")
                print(f"i: {i} j: {j} : {dist[i, j]}")
                if new_distance < dist[i, j]:
                    dist[i, j] = new_distance
                    pred[i, j] = pred[k, j]
    
    print("Distance Matrix After:\n", dist)
    return dist, pred    
    