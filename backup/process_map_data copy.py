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
cafe_dict = {}
nearest_node_cafes = {} # Node osmid : idx for shortest path matrix
updated_nodes = []
updated_edges = []

def process_map_data():
    ####Open Street Map Code###
    # pull map data of Fullerton, CA
    place = {"city": "Fullerton", "state": "California", "country": "USA"}
    G = ox.graph_from_place(place, network_type="drive", truncate_by_edge=True)

    # Fetch features (geometries) for cafes in the specified place
    cafes_geometries = ox.features_from_place(place, tags={"amenity": ["cafe", "coffee_shop"]})

    # Get the nodes in G that correspond to the cafes_geometries
    cafe_nodes = []
    my_dict = dict()
    for cafe_geometry in cafes_geometries.itertuples():
        cafe_name = getattr(cafe_geometry, 'name', None)
        centroid_x, centroid_y = cafe_geometry.geometry.centroid.xy
        nearest_node = ox.distance.nearest_nodes(G, centroid_x[0], centroid_y[0])
        print(f"cafe_name ID: {cafe_name}")
        print(f"nearest node ", nearest_node)
        if not pd.isna(cafe_name):
            cafe_nodes.append(nearest_node) # osmId
            my_dict[nearest_node] = cafe_name

    # Create a subgraph containing only the cafe nodes and their edges
    cafe_subgraph = G.subgraph(cafe_nodes)

    cafe_subgraph = G.subgraph(cafe_nodes + list(G.edges(cafe_nodes)))

    # Create a geolocator instance
    geolocator = Nominatim(user_agent="my_geocoder")
    #Print node data and all edges for each node in cafe_subgraph
    for node in cafe_subgraph.nodes(data=True):
        node_id, node_data = node
        node_data.name = my_dict[node_id]
        print(f"Node ID: {node_id}")
        print(f"Node Data: {node_data}")
        print(f"name {my_dict[node_id]}")
        location = geolocator.reverse((node_data['y'], node_data['x']), language='en')
        print(location.address)
        # Print all edges for the current node
        for edge in cafe_subgraph.edges(node_id, data=True):
            print(f"Edge: {edge}")

        print("\n")

def initializeDistMatrix(cafe_subgraph, G):
    index_mapping = {convert_index: i for i, convert_index in enumerate(cafe_subgraph.nodes)}
    swapped_index_mapping = {value: key for key, value in index_mapping.items()}
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

    printMatrix(dist, num_vertices)

    floyd_warshall(dist)

def printMatrix(dist, num_vertices):
    for row in range(num_vertices):
        for col in range(num_vertices):
            print(f"row {row}, col {col}, len {dist[row][col]}")

    # shortest_path = nx.shortest_path(G, 1853024624, 1853024624, weight='length')
    # ox.plot_graph_route(G, shortest_path, route_color='r', route_linewidth=6, node_size=0, bgcolor='k')

    # shortest_path = nx.shortest_path(G, 2325177846, 1853024624, weight='length')
    # ox.plot_graph_route(G, shortest_path, route_color='r', route_linewidth=6, node_size=0, bgcolor='k')

    # return

    #Print node data and all edges for each node in cafe_subgraph
    for node in cafe_subgraph.nodes(data=True):
        node_id, node_data = node
        print(f"Node ID: {node_id}")
        print(f"Node Data: {node_data}")
        print(f"name {my_dict[node_id]}")
        location = geolocator.reverse((node_data['y'], node_data['x']), language='en')
        print(location.address)
        # Print all edges for the current node
        for edge in cafe_subgraph.edges(node_id, data=True):
            print(f"Edge: {edge}")

        print("\n")

    
    # get building information
    # fullerton_cafes = ox.features_from_place(place,
    #                                              tags={"amenity": "cafe"})

    # for cafe_node in fullerton_cafes:
    #     if cafe_node in G.nodes and G.nodes[cafe_node]:
    #         print(f"Node {cafe_node} has data: {G.nodes[cafe_node]}")
    #     else:
    #         print(f"Node {cafe_node} does not have data")

    # return
    # Create a new graph containing only cafe nodes and their connected edges
    G_cafes = nx.Graph()
    # Add cafe nodes to the new graph
    # for cafe_node in cafe_nodes:
    #     G_cafes.add_node(cafe_node, **G.nodes[cafe_node])

    # for node, data in G_cafes.nodes(data=True):
    #     print(f"Node: {node}, Attributes: {data}")
    # print("end")
    # return
    # Add edges between cafe nodes
    # for u, v, data in G.edges(data=True):
    #     if u in cafe_nodes and v in cafe_nodes:
    #         G_cafes.add_edge(u, v, **data)

    # get building information
    # fullerton_cafes = ox.features_from_place(place,
    #                                              tags={"amenity": "cafe"})

    # G_simplify = simplify_original_graph(G)

    # populate dictionary of cafes in Fullerton
    # create_cafe_dict(cafe_dict, fullerton_cafes)
    # print(cafe_dict['Starbucks'])
    # print(cafe_dict['7 Leaves Cafe'])

    # #example of getting the nearest_node
    # orig = 0
    # orig = ox.nearest_nodes(G, 33.8602673,-117.942165 , return_dist=False)
    # dest = 0
    # dest = ox.nearest_nodes(G, 33.8747911, -117.8900264,return_dist=False)
    # print(orig)
    # print(dest)



    # create an dictionary to convert index (original_index, convert_index)
    index_mapping = {convert_index: i for i, convert_index in enumerate(cafe_subgraph.nodes)}

    print(f"index_mapping ", index_mapping)

    file_path = "D://dist.npy" # no need to re-run algorithm except any special requests

    if os.path.exists(file_path):
        dist = np.load(file_path)
    else:
        dist, pred = floyd_warshall(cafe_subgraph, index_mapping)
        np.save(file_path, dist)

    print(type(index_mapping))
    shortest_paths = get_shortest_paths(cafe_subgraph, dist)

    # Example using built it shortest path function
    origin = 4704306820
    dest = 2757729032

    original_shortest_path = get_shortest_path(origin, dest, shortest_paths, index_mapping, pred)
    print(f"original shortest path: {original_shortest_path}")

    #fig, ax = ox.plot_graph_route(G, route, route_color='r', route_linewidth=6,
    #                              node_size=0, bgcolor='k')
    plt.show()

    return None

def nearest_node_cafes(nodes, cafe_dict):
  num_vertices = len(nodes)

  for i in range(num_vertices):
    nodeid = nodes.iloc[i].name
    idx = i
    #nodeid_dict[nodeid] = idx

  return

# populates cafe_dict structure
def create_cafe_dict(cafe_dict, fullerton_cafes):
    num_cafes = len(fullerton_cafes)

    for i in range(num_cafes):
        name = fullerton_cafes.iloc[i]['name']
        if not pd.isna(name):  # Ignoring features that do not have a name
            osmid = fullerton_cafes.iloc[i].name[1]
            coordinates = fullerton_cafes.iloc[i]['geometry']  # get coordinates
            if type(coordinates) == shapely.geometry.polygon.Polygon: #
                # Ignoring features that have multiple locations
                continue
            else:
                cafe_dict[name] = [osmid, coordinates]
        else:
            continue

    # extract coordinates and repopulate dictionary
    for key in cafe_dict:
        id = cafe_dict[key][0]
        coordinates = cafe_dict[key][1]  # get coordinates
        coordinates = list(
            coordinates.coords)  # get coordinates Geometry is backwards
        Y = coordinates[0][0]
        X = coordinates[0][1]
        cafe_dict[key] = [id, X, Y]


    return cafe_dict

def get_shortest_paths(G, dist):
    num_nodes = len(G.nodes)
    shortest_paths = {}
    for i in range(num_nodes):
        shortest_paths[i] = {}
        for j in range(num_nodes):
            shortest_paths[i][j] = dist[i, j]
    return shortest_paths

# need to re-check
def get_shortest_path(orgin, dest, shortest_paths, index_mapping, pred):
    #global index_mapping
    mapped_orig, mapped_dest = index_mapping[orgin], index_mapping[dest]
    shortest_path_length = shortest_paths[mapped_orig][mapped_dest]
    if np.isinf(shortest_path_length):
        return None
    
    i = dest
    path = []
    while i != orgin:
        path.append(i)
        i = pred[orgin, i]
    path.append(orgin)
    path.reverse()
    return path

# convert string input from UI then convert to numeric value
def convertLocation2Numeric(source, destination):
    return source, destination

# based on the list to simplify the original graph
def simplify_original_graph(graph):
    return graph