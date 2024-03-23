"""
GROUP 1: Anunay Amrit, Angelica Cabato, Pranav Vijay Chand, Riya Chapatwala, Sai Satya Jagannadh Doddipatla, Nhat Ho

Dr. Shah

CPSC 535: Advanced Algorithms (Spring 2024)

"""
# Reference: https://www.programiz.com/dsa/floyd-warshall-algorithm

import numpy as np

def floyd_warshall(graph):
    # get unique vertices
    vertices = list(set([edge[0] for edge in graph]))
    num_vertices = len(vertices)
    # print("vertices are: ", vertices)

    # initialize graph
    dist = np.matrix(np.ones((num_vertices, num_vertices)) * np.inf)
    print(dist)

    # update matrix with vertices and edges
    for row in graph:
        u = row[0] - 1  # start_vertex
        v = row[1] - 1  # end_vertex
        weight = row[2]
        dist[u, v] = weight

    print("Distance Matrix Before:\n", dist)

    # iterate over the graph and update the matrix if new shortest path is found
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    print("Distance Matrix After:\n", dist)

    return dist