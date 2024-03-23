

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
import random

index_mapping = {}
dist = []
final_dist = []
cafe_dict = {}
updated_nodes = []
updated_edges = []
G = []
next_path = []
build_coffe_graph = []
shortest_path_blockages = []
final_nodes = []

def buildmap():
    # Define the place you are interested in (you can customize this)
    place = "Fullerton, California, USA"
    # Fetch the road network data using OSMnx
    G = ox.graph_from_place(place, network_type="drive", simplify=True)
    num_vertices = len(G.nodes)
    print(f"num vertices ", num_vertices)
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
    newInitializeDistMatrix(cafe_subgraph, G)

def newInitializeDistMatrix(cafe_subgraph, G):
    global index_mapping, next_path, final_nodes
    global final_dist, build_coffe_graph
    index_mapping = {convert_index: i for i, convert_index in enumerate(cafe_subgraph.nodes)} # osmId, matrix_id
    swapped_index_mapping = {value: key for key, value in index_mapping.items()} # matrix_id, osmId
    num_vertices = len(cafe_subgraph.nodes)

    for row in range(num_vertices):
        origin_source = swapped_index_mapping[row] # osmid
        for col in range(num_vertices):
            origin_dest = swapped_index_mapping[col] # osmid
            # can manually find any path between 2 nodes based on G - like collecting data
            # Whenever using the simple path => after all, need to rebuild the new graph
            # to plot. Now just make life easier to use the built-in function
            any_path = nx.shortest_path(G, origin_source, origin_dest, weight='length')
            if len(any_path) > 1:
                final_nodes += any_path

    # prepared for a connected graph based on cafe locations and their related nodes
    final_nodes = list(dict.fromkeys(final_nodes))
    num_vertices = len(final_nodes)
    # print(f"final_nodes {final_nodes}")
    # print(f"num vertices {num_vertices}")

    dist = [[np.inf] * num_vertices for _ in range(num_vertices)]
    next_path = [[-1] * num_vertices for _ in range(num_vertices)]
    for i in range(num_vertices):
        dist[i][i] = 0 # Set diagonal elements to 0

    for row in range(num_vertices):
        origin_source = final_nodes[row] # osmid
        for col in range(num_vertices):
            origin_dest = final_nodes[col] # osmid
            if row != col:
                if G.has_edge(origin_source, origin_dest):
                    edge_data = G[origin_source][origin_dest]
                    length = edge_data[0].get('length', np.inf)
                    dist[row][col] = length
                    next_path[row][col] = col

    print(f"dist {dist}")
    #printMatrix(dist, num_vertices)

    final_dist, next_path = floyd_warshallblockages(dist, next_path)
    print(f"dist {final_dist}")
    return

def new_get_shortest_path(osmOrginID, osmDestID):
    global index_mapping, G, build_coffe_graph, shortest_path_blockages
    global final_dist, next_path, final_nodes
    start = final_nodes.index(osmOrginID)
    end = final_nodes.index(osmDestID)

    if (next_path[start][end] == -1):
        return {}
    
    path = [start]
    while (start != end):
        start = next_path[start][end]
        path.append(start)
    
    for i in range(len(path)):
        path[i] = final_nodes[path[i]]

    print(f"path {path}")
    ox.plot_graph_route(G, path, route_color='r', route_linewidth=6, node_size=0, bgcolor='k')
#################################################################################################################
def process_map_data():
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
    initializeDistMatrix(cafe_subgraph, G)

    return G

# collect distance data for all coffe locations
def initializeDistMatrix(cafe_subgraph, G):
    global index_mapping, next_path
    global final_dist, build_coffe_graph
    index_mapping = {convert_index: i for i, convert_index in enumerate(cafe_subgraph.nodes)} # osmId, matrix_id
    swapped_index_mapping = {value: key for key, value in index_mapping.items()} # matrix_id, osmId
    num_vertices = len(cafe_subgraph.nodes)
    dist = [[np.inf] * num_vertices for _ in range(num_vertices)]
    next_path = [[-1] * num_vertices for _ in range(num_vertices)]
    for i in range(num_vertices):
        dist[i][i] = 0 # Set diagonal elements to 0

    for row in range(num_vertices):
        origin_source = swapped_index_mapping[row] # osmid
        for col in range(num_vertices):
            origin_dest = swapped_index_mapping[col] # osmid
            shortest_path_len = nx.shortest_path_length(G, origin_source, origin_dest, weight='length')
            dist[row][col] = shortest_path_len
            next_path[row][col] = col

    #printMatrix(dist, num_vertices)
    build_coffe_graph = buildgraphBasedonInitialDist(dist)
    if 'crs' not in build_coffe_graph.graph:
        build_coffe_graph.graph['crs'] = 'EPSG:4326'

    #print(f"coffe graph ", build_coffe_graph)
    final_dist, next_path = floyd_warshall(build_coffe_graph, next_path, index_mapping)
    # nx.draw(build_graph, with_labels=True, font_weight='bold', arrows=True)
    # plt.show()

def buildgraphBasedonInitialDist(dist):
    global index_mapping
    swapped_index_mapping = {value: key for key, value in index_mapping.items()} # matrix_id, osmId
    #print(f"index_mapping {index_mapping}")
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
    print(f"path from built-in {shortest_path}")
    ox.plot_graph_route(G, shortest_path, route_color='r', route_linewidth=6, node_size=0, bgcolor='k')

def get_shortest_path(osmOrginID, osmDestID):
    global index_mapping, G, build_coffe_graph, shortest_path_blockages
    swapped_index_mapping = {value: key for key, value in index_mapping.items()} # matrix_id, osmId
    global final_dist, next_path
    start = index_mapping[osmOrginID]
    end = index_mapping[osmDestID]

    #print(f"start ", start)
    #print(f"end ", end)

    if (next_path[start][end] == -1):
        return {}
    
    path = [start]
    while (start != end):
        start = next_path[start][end]
        path.append(start)

    shortest_path_nodes = [swapped_index_mapping[index] for index in path]

    # print(f"shortestpath ", shortest_path_nodes)

    #fig, ax = ox.plot_graph(build_coffe_graph, show=False, close=False, edge_color='gray', edge_alpha=0.7)
    #route_color = ox.plot.get_random_color()
    #ox.plot_graph_route(build_coffe_graph, shortest_path_nodes, route_color=route_color, route_linewidth=5, show=False, close=False, edge_color=route_color, edge_linewidth=2)

    # recall the shortest path based on big graph
    final_path = []
   #print(f"shortest_path_blockages", shortest_path_blockages)
    for index in range(len(shortest_path_nodes) - 1):
        startOSM = shortest_path_nodes[index]
        endOSM = shortest_path_nodes[index+1]
        shortest_path = nx.shortest_path(G, startOSM, endOSM, weight='length')
        final_path += shortest_path
    
    final_path = list(dict.fromkeys(final_path))

   # print(f"shortest_path after recall from big graph", final_path)
    #ox.plot_graph_route(G, shortest_path_blockages, route_color='r', route_linewidth=6, node_size=0, bgcolor='k')
    # ox.plot_graph_route(G, final_path, route_color='r', route_linewidth=6,
    # node_size=0, bgcolor='k')
    return final_path

def getpath(startindex, endindex, nextmatrix):
    if next_matrix[start_index][end_index] == -1:
        return []
    path = [start_index]
    while start_index != end_index:
        start_index = next_matrix[start_index][end_index]
        path.append(start_index)
    return path

def simulate_blockages(blockages, index_mapping):
    global final_dist, next_path
    for blockage in blockages:
        source, destination = blockage  # osmID
        source_index = index_mapping[source]
        dest_index = index_mapping[destination]

        # Simulate blockage by setting the distance to infinity
        final_dist[source_index][dest_index] = float('inf')
        final_dist[dest_index][source_index] = float('inf')  # If undirected graph

    # Re-run Floyd-Warshall to update the distance matrix with blockages
    final_dist, next_path = floyd_warshallblockages(final_dist, next_path)
"""
def draw_updated_path(G, start_osm_id, end_osm_id, index_mapping, next_matrix):
    # Get the indexes for the start and end nodes
    start_index = index_mapping[start_osm_id]
    end_index = index_mapping[end_osm_id]

    # Retrieve the path using the updated 'next' matrix
    path_indexes = get_path(start_index, end_index, next_matrix)
    osm_path = [swapped_index_mapping[index] for index in path_indexes]  # Convert indexes back to osm IDs

    # Draw the path on the graph
    fig, ax = ox.plot_graph_route(G, osm_path, route_color='r', route_linewidth=6, node_size=0)
"""
def updateDictforBlockages(blockages):
    global index_mapping
    global final_dist, next_path
    simulate_blockages(blockages, index_mapping)

    return

def simulate_blockages(blockages, index_mapping):
    global final_dist, next_path
    global shortest_path_blockages
    for blockage in blockages:
        source, destination = blockage #osmID
        source_index = index_mapping[source]
        dest_index = index_mapping[destination]

        #print(f"source_index {source_index}")
        #print(f"dest_index {dest_index}")

        final_dist[source_index][dest_index] = float('inf')  # simulate blockage
        shortest_path_blockages = nx.shortest_path(G, source, destination, weight='length')
        #ox.plot_graph_route(G, shortest_path_blockages, route_color='r',
        # route_linewidth=6, node_size=0, bgcolor='k')
        # print(f"shortest_path blockages {shortest_path_blockages}")
        # for osm in shortest_path:
        #     dest_index = index_mapping[osm]
        #     final_dist[source_index][dest_index] = float('inf')  # simulate blockage

    # rerun floyd-warshall algo to update dist matrix
    final_dist, next_path = floyd_warshallblockages(final_dist, next_path)

def implement_blockage(source_osm_id, dest_osm_id, cur_shortest_path):
    global index_mapping

    k = 10  # number of routes to generate
    # Get unique route options
    possible_routes = list(ox.k_shortest_paths(G, source_osm_id,
                                               dest_osm_id, k,
                                               weight='length'))


    # remove current shortest path from list, as we need to return a new path
    indices = [i for i, x in enumerate(possible_routes) if x == cur_shortest_path]

    for i in range(len(indices)):
        possible_routes.pop(i)

    # If there are no other possible paths, return error
    if not possible_routes:
        print("No possible path between nodes due to blockage.")
        return

    # return another route
    updated_path_after_blockage = None

    rand_path_index = random.randint(0, len(possible_routes) - 1)
    rand_path = possible_routes[rand_path_index]
    updated_path_after_blockage = rand_path
    #print(f'Updated path {rand_path_index}:', updated_path_after_blockage)

    return updated_path_after_blockage
