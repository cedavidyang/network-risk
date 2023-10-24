#%%
# Import Libraries
import osmnx as ox
import networkx as nx
import numpy as np
from highway_network import generate_highway_graph
from highway_network import _append_compute_attributes
from highway_network import compute_capacity
from highway_network import identify_end_nodes, generate_od_pairs


#%%
# # download raw graph 
# # download/load and generate raw osmnx Graph
# place = {"state": "Oregon", "country": "USA"}
# filter = '["highway"~"motorway|trunk|primary"]'
# G_raw = ox.graph_from_place(place, truncate_by_edge=True, custom_filter=filter)

# # save the graph for future use
# ox.save_graphml(G_raw, filepath="./assets_2/or_hw_raw.graphml")

G_raw = ox.load_graphml(filepath="./assets_2/or_hw_raw.graphml", )

# fig, ax = ox.plot_graph(G_raw)
# %%
# load the graph
min_lane, max_speed = 2, 60.0
# G_raw = ox.load_graphml(filepath="./assets_2/or_hwy2prim_raw.graphml")
G_raw = ox.project_graph(G_raw, to_crs="EPSG:3857")

G_attr = _append_compute_attributes(G_raw, min_lane=min_lane, max_speed=max_speed,
                               inplace=False)


# %%


G_attr = G_raw.copy()
bridge_indx = 0
for u, v, k, data in G_attr.edges(keys=True, data=True):
    if 'lanes' not in data:
        lane = min_lane
    elif isinstance(data['lanes'], list):
        lane = np.min([int(l[0]) for l in data['lanes']])
    elif isinstance(data['lanes'], str):
        lane = int(data['lanes'])
    else:
        lane = min_lane

    if 'maxspeed' not in data:
        speed = max_speed
    elif isinstance(data['maxspeed'], list):
        speed = np.max([int(l[:2]) for l in data['maxspeed']])
    elif isinstance(data['maxspeed'], str):
        speed = int(data['maxspeed'][:2])
    else:
        speed = max_speed

    # capacity = lane/max_speed*speed   *** how is affected when graph is not simplified?

    capacity = compute_capacity(lane, speed,
                                min_lane=min_lane, max_speed=max_speed)

    if 'bridge' not in data:
        bridge_id = 0
    elif isinstance(data['bridge'], list) and ('yes' in data['bridge']):
        bridge_indx += 1
        bridge_id = bridge_indx
    elif data['bridge'].lower() == 'yes':
        bridge_indx += 1
        bridge_id = bridge_indx
    else:
        bridge_id = 0
    G_attr.edges[u,v,k].update({'lane': lane,
                           'speed': speed,
                            'capacity': capacity,
                            'bridge_id': bridge_id})
# Convert to DiGraph from MultiDiGraph keep edge with bridge else the shortest length

G_dir = ox.get_digraph(G_attr, weight= 'length')

# # node pairs with multiple edges are:
# pairs = [(u, v) for u, v, k in G_attr.edges(keys=True) if k != 0]

# # which k to keep: the one with bridge and/or the shortest length
# keep_k = []
# for u, v in pairs:
#     k_list = [k for k in G_attr[u][v].keys()]
#     k_lengh = [G_attr[u][v][k]['length'] for k in k_list]
#     k_how_many_bridges = len([k for k in k_list if G_attr[u][v][k]['bridge_id'] != 0])
#     # keep k with maximum number of bridges
#     if k_how_many_bridges == 0:
#         keep_k.append(k_list[np.argmin(k_lengh)])
#     else:
#         keep_k.append(k_list[np.argmax(k_how_many_bridges)])


# # remove the edges with k not in keep_k
# for u, v in pairs:
#     k_list = [k for k in G_attr[u][v].keys()]
#     for k in k_list:
#         if k not in keep_k:
#             G_attr.remove_edge(u, v, k)

# # now convert to DiGraph
# G_dir = nx.DiGraph(G_attr).copy()


# save to a geopackage for GIS postprocessing
ox.save_graph_geopackage(G_attr, filepath="./assets_2/or_hw_atrr.gpkg")
ox.save_graph_geopackage(G_dir, filepath="./assets_2/or_hw_dir.gpkg")
# %%
G_comp = nx.Graph()
G_comp.add_nodes_from(G_dir.nodes)



for u, v, data in G_dir.edges(data=True):
    lane = int(data['lane'])
    speed = float(data['speed'])
    capacity = float(data['capacity'])
    bridge = int(data['bridge_id'])
    length = float(data['length'])
    G_comp.add_edge(int(u), int(v),  lane=lane, speed=speed,
                    capacity=capacity, length=length, bridge_id=bridge)


# %%
# save the graph for future use
G_comp = nx.MultiDiGraph(G_comp)
ox.save_graphml(G_comp, filepath="./assets_2/or_hw_comp.graphml")
# %%

# identify end nodes and generate od pairs
bnd_nodes, end_nodes = identify_end_nodes(G_dir, state_polygon= './assets_2/OR_state_boundary/OR_state.shp')
bnd_od, shortest_path_log = generate_od_pairs(
    G_dir, end_nodes=bnd_nodes, min_distance=5e3)


# save the OD pairs to assets
np.savez('./assets_2/bnd_od.npz', bnd_od=bnd_od)
