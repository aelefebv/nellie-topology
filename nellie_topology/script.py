import numpy as np
import tifffile
from scipy.ndimage import label
import csv
import os
import networkx as nx

pixel_class_path = r"D:\test_files\Nellie Topological properties\507\nellie_test\nellie_output\507-ome.ome-ch0-im_pixel_class.ome.tif"
visualize = False

save_name = os.path.basename(pixel_class_path).split("-")[0] + "_adjacency_list.csv"
save_path = os.path.join(os.path.dirname(pixel_class_path), save_name)

if visualize:
    import napari
    viewer = napari.Viewer()

# Load your 3D TIF
skeleton = tifffile.imread(pixel_class_path)  
viewer.add_image(skeleton) if visualize else None

struct = np.ones((3,3,3))
# Get trees
trees, num_trees = label(skeleton>0, structure=struct)
viewer.add_labels(trees) if visualize else None

# Convert tips and lone-tips to nodes, junctions are already nodes
skeleton[skeleton == 2] = 4
skeleton[skeleton == 1] = 4

# Remove all voxels == 4 (nodes)
no_nodes = np.where(skeleton == 4, 0, skeleton)
edges, num_edges = label(no_nodes>0, structure=struct)
viewer.add_labels(edges) if visualize else None

# nodes only
nodes = np.where(skeleton == 4, 4, 0)
node_labels, num_nodes = label(nodes>0, structure=struct)
viewer.add_labels(node_labels) if visualize else None

# Build adjacency: which edges connect to which node?
node_edges = {}

# Loop over each distinct node label
for j_id in range(1, num_nodes + 1):
    # Collect all the voxels that belong to this node
    j_coords = np.argwhere(node_labels == j_id)
    
    # A set to accumulate edge IDs that connect to this node
    connected_edges = set()
    
    # For each voxel in this node, check all neighbors in a 3x3x3 region
    for (x, y, z) in j_coords:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        # Skip the center voxel itself
                        continue
                    xx, yy, zz = x + dx, y + dy, z + dz
                    # Make sure we stay within image bounds
                    if (0 <= xx < skeleton.shape[0] and
                        0 <= yy < skeleton.shape[1] and
                        0 <= zz < skeleton.shape[2]):
                        # Check if this neighbor belongs to a edge
                        edge_label = edges[xx, yy, zz]
                        if edge_label != 0:
                            connected_edges.add(edge_label)
    
    # Store the set of connected edge labels for this node
    node_edges[j_id] = connected_edges

edge_nodes = {}  # key: edge_label, value: set of node_labels

for n_id, e_set in node_edges.items():
    for e_id in e_set:
        if e_id not in edge_nodes:
            edge_nodes[e_id] = set()
        edge_nodes[e_id].add(n_id)

G = nx.Graph()
for j_id in range(1, num_nodes+1):
    G.add_node(j_id)

# Add edges between nodes
for e_id, connected_nodes in edge_nodes.items():
    cn = list(connected_nodes)
    if len(cn) == 2:
        n1, n2 = cn
        G.add_edge(n1, n2, edge_id=e_id)
    elif len(cn) == 1:
        # Self loop possibility
        (n1,) = cn
        G.add_edge(n1, n1, edge_id=e_id)
    elif len(cn) > 2:
        # Possibly connect all pairs to keep adjacency consistent
        for i in range(len(cn)):
            for j in range(i+1, len(cn)):
                G.add_edge(cn[i], cn[j], edge_id=e_id)
                
if visualize:
    import matplotlib.pyplot as plt

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color="r")
    nx.draw_networkx_labels(G, pos, font_family="sans-serif", font_size=8)
    nx.draw_networkx_edges(G, pos, edge_color="b", width=3)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# To look at individual trees:
components = nx.connected_components(G)

# Open a new CSV file to write
with open(save_path, "w", newline="") as f:
    writer = csv.writer(f)
    # Write the header row
    writer.writerow(["component_num", "node", "adjacencies"])
    
    # Enumerate all components, starting at 1
    for comp_num, comp in enumerate(components, start=1):
        # Create a subgraph for the current component
        subG = G.subgraph(comp).copy()
        
        # For each node in the subgraph, write out its adjacencies
        for node in sorted(subG.nodes()):
            # Get the neighbors/adjacencies of this node
            adjacency = sorted(list(subG[node]))
            writer.writerow([comp_num, node, adjacency])

print(f"Adjacency list CSV saved to {save_path}.")
