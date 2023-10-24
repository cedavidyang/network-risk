#%%
# Import Libraries
import osmnx as ox
import networkx as nx
import numpy as np
from highway_network import generate_highway_graph
from highway_network import _append_compute_attributes, generate_compute_graph
from highway_network import compute_capacity
from highway_network import identify_end_nodes, generate_od_pairs

from highway_risk_temper import net_capacity


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

G_comp = generate_compute_graph(G_attr)


# %%
# save the graph for future use
G_save = nx.MultiDiGraph(G_comp)
ox.save_graphml(G_save, filepath="./assets_2/or_hw_comp.graphml")
G_save_attr = nx.MultiDiGraph(G_attr)
ox.save_graph_geopackage(G_save_attr, filepath="./assets_2/or_hw_attr.gpkg", directed=True)
# %%

# identify end nodes and generate od pairs
bnd_nodes, end_nodes = identify_end_nodes(G_attr, state_polygon= './OR_state_boundary/OR_state.shp')
bnd_od, shortest_path_log = generate_od_pairs(
    G_attr, end_nodes=bnd_nodes, min_distance = 5e3, max_distance= 1e13)

# save the OD pairs to assets
np.savez('./assets_2/bnd_od.npz', bnd_od=bnd_od)

#%%
max_flow = net_capacity(G_comp, od_pairs=bnd_od, capacity='capacity')
# %%
# n = 0
# for u, v, data in G_comp.edges(data=True):
#     if data['bridge_id'] != 0:
#         n += 1
# print(n)
# %%
# G_attr.nodes[85199866]
# {'y': 5159803.924194695, 'x': -13772330.404158983, 'street_count': 1, 'lon': -123.718949, 'lat': 41.992152}
# G_attr.nodes[1223727013]
# {'y': 5720064.94554994, 'x': -13655843.974672744, 'street_count': 1, 'lon': -122.6725336, 'lat': 45.6225337}

# %%

