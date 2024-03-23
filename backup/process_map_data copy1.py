"""
GROUP 1: Anunay Amrit, Angelica Cabato, Pranav Vijay Chand, Riya Chapatwala, Sai Satya Jagannadh Doddipatla, Nhat Ho

Dr. Shah

CPSC 535: Advanced Algorithms (Spring 2024)

"""

# Reference Option #1: https://www.geeksforgeeks.org/working-with-geospatial
# -data-in-python/

# Reference Option #2: https://networkx.org/documentation/stable/auto_examples/geospatial/plot_osmnx.html

###### Imports needed for Open Street Map API ######
import networkx as nx
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from floyd_warshall import floyd_warshall
import numpy as np
import os
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon
from geopy.geocoders import Nominatim
import math

index_mapping = {}
dist = []
final_dist = []
cafe_dict = {}
updated_nodes = []
updated_edges = []
G = []

def process_map_data():
    ####Open Street Map Code###
    # pull map data of Fullerton, CA
    global G
    place = {"city": "Fullerton", "state": "California", "country": "USA"}
    G = ox.graph_from_place(place, network_type="drive", truncate_by_edge=True)

    # Fetch features (geometries) for cafes in the specified place
    cafes_geometries = ox.features_from_place(place, tags={"amenity": ["cafe", "coffee_shop"]})

    # Get the osmID for each coffe node
    cafe_nodes = []
    my_dict = dict()
    for cafe_geometry in cafes_geometries.itertuples():
        cafe_name = getattr(cafe_geometry, 'name', None)
        centroid_x, centroid_y = cafe_geometry.geometry.centroid.xy
        nearest_node = ox.distance.nearest_nodes(G, centroid_x[0], centroid_y[0])
        if not pd.isna(cafe_name):
            cafe_nodes.append(nearest_node) # osmId
            my_dict[nearest_node] = cafe_name
    # Create a subgraph containing only the cafe nodes and their edges
    cafe_subgraph = G.subgraph(cafe_nodes)

    # Create a geolocator instance
    # geolocator = Nominatim(user_agent="my_geocoder")
    #Print node data and all edges for each node in cafe_subgraph
    # for node in cafe_subgraph.nodes(data=True):
    #     node_id, node_data = node
    #     location = geolocator.reverse((node_data['y'], node_data['x']), language='en')
        # Print all edges for the current node
        # for edge in cafe_subgraph.edges(node_id, data=True):
        #     print(f"Edge: {edge}")
        # print("\n")

    initializeDistMatrix(cafe_subgraph, G)

def initializeDistMatrix(cafe_subgraph, G):
    global index_mapping
    global final_dist
    index_mapping = {convert_index: i for i, convert_index in enumerate(cafe_subgraph.nodes)} # osmId, matrix_id
    swapped_index_mapping = {value: key for key, value in index_mapping.items()} # matrix_id, osmId
    num_vertices = len(cafe_subgraph.nodes)
    dist = [[np.inf] * num_vertices for _ in range(num_vertices)]
    # Set diagonal elements to 0
    for i in range(num_vertices):
        dist[i][i] = 0

    for row in range(num_vertices):
        origin_source = swapped_index_mapping[row] # osmid
        for col in range(num_vertices):
            origin_dest = swapped_index_mapping[col] # osmid
            shortest_path_len = nx.shortest_path_length(G, origin_source, origin_dest, weight='length')
            dist[row][col] = shortest_path_len

    #printMatrix(dist, num_vertices)
    final_dist = dist

def printMatrix(dist, num_vertices):
    for row in range(num_vertices):
        for col in range(num_vertices):
            print(f"row {row}, col {col}, len {dist[row][col]}")

# need to re-check
def get_shortest_path_builtin(osmOrginID, osmDestID):
    global G
    shortest_path = nx.shortest_path(G, osmOrginID, osmDestID, weight='length')
    ox.plot_graph_route(G, shortest_path, route_color='r', route_linewidth=6, node_size=0, bgcolor='k')

def get_shortest_path():
    return

def updateDictforBlockages(source, dest):
    global index_mapping
    global final_dist
    row = index_mapping[source]
    col = index_mapping[dest]
    final_dist[row][col] = float('inf')
    # rerun floyd-warshall algo to update dist matrix
    final_dist = floyd_warshall(final_dist)
    get_shortest_path()
    get_shortest_path_builtin(source, dest)
    return