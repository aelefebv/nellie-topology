#%%
import numpy as np
import tifffile
from scipy.ndimage import label, generate_binary_structure

#%%
import napari
viewer = napari.Viewer()

#%%
pixel_class_path = r"D:\test_files\Nellie Topological properties\500\nellie_test\nellie_output\500-ome.ome-ch0-im_pixel_class.ome.tif"

# Load your 3D TIF
skeleton = tifffile.imread(pixel_class_path)  
viewer.add_image(skeleton)
# %%
struct = np.ones((3,3,3))
# Get trees
trees, num_trees = label(skeleton>0, structure=struct)
viewer.add_labels(trees)

# Convert tips and lone-tips to nodes, junctions are already nodes
skeleton[skeleton == 2] = 4
skeleton[skeleton == 1] = 4

# Remove all voxels == 4 (nodes)
no_nodes = np.where(skeleton == 4, 0, skeleton)
edges, num_edges = label(no_nodes>0, structure=struct)
viewer.add_labels(edges)

# nodes only
nodes = np.where(skeleton == 4, 4, 0)
node_labels, num_nodes = label(nodes>0, structure=struct)
viewer.add_labels(node_labels)

# %%
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


# %%
import networkx as nx

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
                
adjacency_list = {node: list(G[node]) for node in G.nodes()}

# Get adjacency matrix
from networkx.convert_matrix import to_numpy_array
A = to_numpy_array(G, nodelist=sorted(G.nodes()))

# Detect cycles
cycles = nx.cycle_basis(G)

print("Adjacency list:")
for node, neighbors in adjacency_list.items():
    print(node, "->", neighbors)

# %%
nx.draw(G)
# %%
