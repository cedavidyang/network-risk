#%%
#    The script performs the following steps:
#    1. Download the raw highway network data from OpenStreetMap.
#    2. Generate the graph with GIS attributes.
#    3. Generate the computational graph.
#    4. Generate the OD pairs.
#    5. Save results to the assets folder.
#    The user does not need to run the script as the results are already provided in the assets folder.
#    The script can be modified to generate data for a different region or with different parameters.

import osmnx as ox
import networkx as nx
import numpy as np
from netrisk_TMCMC.highway_network import generate_highway_graph, generate_compute_graph
from netrisk_TMCMC.highway_network import identify_bnd_nodes, generate_od_pairs


if __name__ == "__main__":
    # parameters
    min_lane, max_speed = 2, 60.0
    multi2simple_weight = 'length'
    crs = 'EPSG:3857'
    min_od_dist = 50e3

    place = {"state": "Oregon", "country": "USA"}
    filter = '["highway"~"motorway|trunk|primary"]'

    or_boundary = './assets/OR-boundary/OR-boundary.shp'
    raw_graph_path = './assets/or_hw_raw'
    gis_graph_path = './assets/or_hw_gis'
    comp_graph_path = './assets/or_hw_comp'
    od_path = './assets/bnd_od'
        
    # download/load and generate raw osmnx Graph
    G_raw = ox.graph_from_place(
        place, truncate_by_edge=True, custom_filter=filter
    )
    # Save the graph for future use
    ox.save_graphml(G_raw, filepath=raw_graph_path+'.graphml')
    ox.save_graph_geopackage(G_raw, filepath=raw_graph_path+'.gpkg')

    # Generate graph with GIS attributes
    G_gis = generate_highway_graph(
        G_raw, crs=crs,
        min_lane=min_lane, max_speed=max_speed,
        multi_weight=multi2simple_weight,
        save_to=gis_graph_path
    )

    # Generate computational graph
    G_comp = generate_compute_graph(G_gis, save_to=comp_graph_path)

    # Identify boundary nodes and generate od pairs
    bnd_nodes, end_nodes = identify_bnd_nodes(
        G_gis, state_polygon=or_boundary
    )
    bnd_od, shortest_path_log = generate_od_pairs(
        G_gis, end_nodes=bnd_nodes,
        min_distance=min_od_dist,
        save_to=od_path,
    )