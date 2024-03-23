from flask import Flask, jsonify, request, render_template
import networkx as nx
import osmnx as ox
from process_map_data import (buildmap, get_shortest_path_builtin, new_get_shortest_path, process_map_data, get_shortest_path, implement_blockage)

app = Flask(__name__)

# Global variables
G = process_map_data()  # Initialize the graph
name_to_node_dict = {
    "Pilgrim's Coffee Shop": 4704306820,
    "Santa Fe Express Cafe": 1391863431,
    "Jay's Coffee Waffles & More": 122562954,
    "Kawaii Boba": 2243492748,
    "Starbucks": 2243492752,
    "Intentional Coffee": 67543057,
    "Starbucks": 8816967697,
    "McClain's Coffeehouse": 122836756,
    "Library Cafe": 5197254171,
    "Philz Coffee": 122757283,
    "The Gastronome": 414550180,
    "Starbucks": 3583764518,
    "Starbucks": 1931779371,
    "Max Bloom's Cafe Noir": 122925745,
    "Sharetea": 4083327025,
    "Starbucks": 2243418547,
    "The Coffee Bean & Tea Leaf": 122817213,
    "Starbucks": 2574034240,
    "Donut Star": 122628045,
    "525 Coffee Co": 122731859,
    "Veronese Gallery and Cafe": 122628569,
    "Coffee Code": 122868068,
    "The Smoking Tiger Coffee and Bread": 2243433066,
    "Starbucks": 67543020,
    "Starbucks": 1853024624,
    "Made Coffee": 2325177846,
    "Eggspresso": 122900858,
    "Dripp": 122757243,
    "The Stinger Cafe": 122728319,
}

@app.route('/')
def index():
    # Render the HTML interface
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/get_cafes', methods=['GET'])
def get_cafes():
    # Return a list of cafes for populating the dropdowns
    cafes = [{"name": name, "lat": G.nodes[node]["y"], "lng": G.nodes[node]["x"]} for name, node in name_to_node_dict.items()]
    return jsonify({"cafes": cafes})

@app.route('/calculate_path', methods=['POST'])

def calculate_path():
    data = request.json
    source_name = data['source']
    destination_name = data['destination']
    source_node = name_to_node_dict.get(source_name)
    destination_node = name_to_node_dict.get(destination_name)

    # Always calculate the shortest path first
    shortest_path = get_shortest_path(source_node, destination_node)

    # Check for blockage status from the request
    blockage = data.get('blockage', False)

    if not blockage:
        # If there's no blockage, return the shortest path
        path_coords = [[G.nodes[node]['y'], G.nodes[node]['x']] for node in shortest_path]
        return jsonify({"path": path_coords})
    else:
        # If there's a blockage, use the shortest path to find an alternative route
        updated_path = implement_blockage(source_node, destination_node, shortest_path)
        # Ensure that implement_blockage returns the updated path correctly
        if updated_path is None:
            # Handle case where no alternate path exists
            return jsonify({"error": "No alternate path exists"}), 400
        path_coords = [[G.nodes[node]['y'], G.nodes[node]['x']] for node in updated_path]
        return jsonify({"path": path_coords})

def main():
    buildmap()
    new_get_shortest_path(1391863431, 122757243)
    get_shortest_path_builtin(1391863431, 122757243)

if __name__ == '__main__':
    main()
    app.run(debug=True)