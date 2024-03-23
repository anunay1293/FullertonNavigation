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
from floyd_warshall import floyd_warshall, floyd_warshallblockages
import numpy as np
import os
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon
from geopy.geocoders import Nominatim
import math
import matplotlib.pyplot as plt

index_mapping = {}
dist = []
final_dist = []
cafe_dict = {}
updated_nodes = []
updated_edges = []
G = []
next = []
build_coffe_graph = []
shortest_path_blockages = []

def buildmap():
    # Define the place you are interested in (you can customize this)
    place = "Fullerton, California, USA"
    # Fetch the road network data using OSMnx
    G = ox.graph_from_place(place, network_type="drive", simplify=True)
    num_vertices = len(G.nodes)
    print(f"num vertices ", num_vertices)

    nodes_to_keep = list(G.nodes)[:10]
    subgraph = G.subgraph(nodes_to_keep)

    for node in nodes_to_keep:
        print(f"node {node}")
        for edge in subgraph.edges(node, data=True):
            print(f"Edge: {edge}")
    
    

def process_map_data():
    ####Open Street Map Code###
    # pull map data of Fullerton, CA
    global G
    place = {"city": "Fullerton", "state": "California", "country": "USA"}
    G = ox.graph_from_place(place, network_type="drive", truncate_by_edge=True)

    # Fetch features (geometries) for cafes in the specified place
    cafes_geometries = ox.features_from_place(place, tags={"amenity": ["cafe", "coffee_shop"]})
    # Get the coffe nodes in G
    cafe_nodes = []
    my_dict = dict() # restore name of osmID node
    for cafe_geometry in cafes_geometries.itertuples():
        cafe_name = getattr(cafe_geometry, 'name', None)
        centroid_x, centroid_y = cafe_geometry.geometry.centroid.xy
        nearest_node = ox.distance.nearest_nodes(G, centroid_x[0], centroid_y[0])
        if not pd.isna(cafe_name):
            cafe_nodes.append(nearest_node) # osmId
            my_dict[nearest_node] = cafe_name

    # Create a temporary graph containing only the cafe nodes and their edges
    cafe_subgraph = G.subgraph(cafe_nodes)
    #printCoffeLocationInfo(cafe_subgraph, my_dict)
    initializeDistMatrix(cafe_subgraph, G)

# collect distance data for all coffe locations
def initializeDistMatrix(cafe_subgraph, G):
    global index_mapping, next
    global final_dist, build_coffe_graph
    index_mapping = {convert_index: i for i, convert_index in enumerate(cafe_subgraph.nodes)} # osmId, matrix_id
    swapped_index_mapping = {value: key for key, value in index_mapping.items()} # matrix_id, osmId
    num_vertices = len(cafe_subgraph.nodes)
    dist = [[np.inf] * num_vertices for _ in range(num_vertices)]
    next = [[-1] * num_vertices for _ in range(num_vertices)]
    for i in range(num_vertices):
        dist[i][i] = 0 # Set diagonal elements to 0

    for row in range(num_vertices):
        origin_source = swapped_index_mapping[row] # osmid
        for col in range(num_vertices):
            origin_dest = swapped_index_mapping[col] # osmid
            shortest_path_len = nx.shortest_path_length(G, origin_source, origin_dest, weight='length')
            dist[row][col] = shortest_path_len
            next[row][col] = col

    #printMatrix(dist, num_vertices)
    build_coffe_graph = buildgraphBasedonInitialDist(dist)
    if 'crs' not in build_coffe_graph.graph:
        build_coffe_graph.graph['crs'] = 'EPSG:4326'

    print(f"coffe graph ", build_coffe_graph)
    final_dist, next = floyd_warshall(build_coffe_graph, next, index_mapping)
    # nx.draw(build_graph, with_labels=True, font_weight='bold', arrows=True)
    # plt.show()

def buildgraphBasedonInitialDist(dist):
    global index_mapping
    swapped_index_mapping = {value: key for key, value in index_mapping.items()} # matrix_id, osmId
    print(f"index_mapping {index_mapping}")
    new_G = nx.DiGraph()
    # Add nodes with their corresponding attributes
    for index, node_id in index_mapping.items():
        new_G.add_node(swapped_index_mapping[node_id]) # add osmID

    # Add edges with their distances
    for i, row in enumerate(dist):
        for j, distance in enumerate(row):
            if i != j and distance != float('inf'):
                new_G.add_edge(swapped_index_mapping[i], swapped_index_mapping[j], length=distance)
    return new_G

def printMatrix(dist, num_vertices):
    for row in range(num_vertices):
        for col in range(num_vertices):
            print(f"row {row}, col {col}, len {dist[row][col]}")

# print coffe location's information
def printCoffeLocationInfo(cafe_subgraph, my_dict):
    # Create a geolocator instance
    geolocator = Nominatim(user_agent="my_geocoder")
    #Print node data and all edges for each node in cafe_subgraph
    for node in cafe_subgraph.nodes(data=True):
        node_id, node_data = node
        location = geolocator.reverse((node_data['y'], node_data['x']), language='en')
        print(f"Node ID: {node_id}")
        print(f"Node Data: {node_data}")
        print(f"name {my_dict[node_id]}")
        print(location.address)
        # Print all edges for the current node
        for edge in cafe_subgraph.edges(node_id, data=True):
            print(f"Edge: {edge}")
        print("\n")

def get_shortest_path_builtin(osmOrginID, osmDestID):
    global G, index_mapping
    shortest_path = nx.shortest_path(G, osmOrginID, osmDestID, weight='length')
    ox.plot_graph_route(G, shortest_path, route_color='r', route_linewidth=6, node_size=0, bgcolor='k')

def get_shortest_path(osmOrginID, osmDestID):
    global index_mapping, G, build_coffe_graph, shortest_path_blockages
    swapped_index_mapping = {value: key for key, value in index_mapping.items()} # matrix_id, osmId
    global final_dist, next
    start = index_mapping[osmOrginID]
    end = index_mapping[osmDestID]

    print(f"start ", start)
    print(f"end ", end)

    if (next[start][end] == -1):
        return {}
    
    path = [start]
    while (start != end):
        start = next[start][end]
        path.append(start)

    shortest_path_nodes = [swapped_index_mapping[index] for index in path]

    print(f"shortestpath ", shortest_path_nodes)
    #fig, ax = ox.plot_graph(build_coffe_graph, show=False, close=False, edge_color='gray', edge_alpha=0.7)
    #route_color = ox.plot.get_random_color()
    #ox.plot_graph_route(build_coffe_graph, shortest_path_nodes, route_color=route_color, route_linewidth=5, show=False, close=False, edge_color=route_color, edge_linewidth=2)

    # recall the shortest path based on big graph
    final_path = []
    print(f"shortest_path_blockages", shortest_path_blockages)
    for index in range(len(shortest_path_nodes) - 1):
        startOSM = shortest_path_nodes[index]
        endOSM = shortest_path_nodes[index+1]
        shortest_path = nx.shortest_path(G, startOSM, endOSM, weight='length')
        final_path += shortest_path
    
    final_path = list(dict.fromkeys(final_path))
    print(f"shortest_path after recall from big graph", final_path)
    #ox.plot_graph_route(G, shortest_path_blockages, route_color='r', route_linewidth=6, node_size=0, bgcolor='k')
    ox.plot_graph_route(G, final_path, route_color='r', route_linewidth=6, node_size=0, bgcolor='k')
    return final_path

def updateDictforBlockages(blockages):
    global index_mapping
    global final_dist, next
    simulate_blockages(blockages, index_mapping, final_dist)
    # rerun floyd-warshall algo to update dist matrix
    final_dist, next = floyd_warshallblockages(final_dist, next)
    return

def simulate_blockages(blockages, index_mapping, final_dist):
    global shortest_path_blockages
    for blockage in blockages:
        source, destination = blockage #osmID
        source_index = index_mapping[source]
        dest_index = index_mapping[destination]

        print(f"source_index {source_index}")
        print(f"dest_index {dest_index}")

        final_dist[source_index][dest_index] = float('inf')  # simulate blockage

        shortest_path_blockages = nx.shortest_path(G, source, destination, weight='length')
        print(f"shortest_path blockages {shortest_path_blockages}")
        # for osm in shortest_path:
        #     dest_index = index_mapping[osm]
        #     final_dist[source_index][dest_index] = float('inf')  # simulate blockage
