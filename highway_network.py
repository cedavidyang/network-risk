# %%
import sys
import networkx as nx
import osmnx as ox
import geopandas as gpd
import numpy as np
import pandas as pd
import itertools

from osmnx import utils_graph
from shapely.ops import linemerge
from shapely.geometry import LineString, MultiLineString
from shapely.geometry import Point


def _append_compute_attributes(G, min_lane=2, max_speed=60, inplace=True):
    """Add numertical attributes needed for graph flow computation and further graph simplification

    Args:
        G (Graph): original graph after preliminary simplification and consolidation
        min_lane (int, optional): minimum number of lanes for a highway link in one direction. Defaults to 2.
        max_speed (int, optional): max speed on a highway link. Defaults to 60.
        inplace (bool, optional): whether to add attributes in place. Defaults to True

    Returns:
        H (MultiDiGraph): Graph with added attributes
    """
    
    if isinstance(G, (nx.Graph, nx.DiGraph)):
        H = G if inplace else G.copy()
    elif G.graph['no_parallel']:
        H = ox.get_undirected(G)
        inplace = False
        print("Warning: inplace has been set to False")
    else:
        sys.exit("G must be Graph or DiGraph or can be converted to such")

    bridge_indx = 0
    for u, v, data in H.edges(data=True):

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

        # capacity = lane/max_speed*speed
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
    
        H.edges[u,v].update({'lane': lane,
                             'speed': speed,
                             'capacity': capacity,
                             'bridge_id': bridge_id,
                             'capacity_param': (min_lane, max_speed)})
        
    if not inplace:
        return H


def _simplify_graph_d2(G, track_merged=False):
    """Further simplify graph based link properties connecting 2-degree nodes

    Args:
        G (Graph): Graph objective with expanded properties
        track_merged (bool, optional): whether track merging info. Defaults to False.

    Returns:
        G (Graph): futher simplified graph
    """

    # STEP 1: find subgraph components including only d2 nodes

    if isinstance(G, nx.DiGraph):
        G = nx.to_undirected(G)
    elif isinstance(G, nx.Graph):
        G = G.copy()
    else:
        sys.exit("G must be DiGraph or Graph")
    
    sub_nodes = []
    for u, d in G.degree():
        if d == 2:
            # find neighboring links (use predecessor and successor for directed Graph)
            bridge_both = []
            for v in G.neighbors(u):
                bridge_both.append(G[u][v]['bridge_id'])
            if bridge_both[0] == bridge_both[1]:
                sub_nodes.append(u)

    G_sub = G.subgraph(sub_nodes)

    # STEP 2: Merge links in subgraph components with (3 or more nodes)
    # the follow snippet is modified based on osmnx.simplification source code

    all_nodes_to_remove = []
    all_edges_to_add = []
    attrs_to_sum = {"length", "travel_time"}
    attrs_to_min = {'lane'}
    attrs_to_max = {'speed'}
    attrs_to_unique = {'bridge_id', 'capacity_param'}
    # attrs_to_merge = {'osmid', 'lanes'}

    for comp in nx.connected_components(G_sub):
        if len(comp) > 2:
            od_pair = [n for n in comp if G_sub.degree(n) == 1]
            path = nx.shortest_path(G_sub, od_pair[0], od_pair[1], weight='length')

            merged_edges = []
            path_attributes = {}
            for u, v in zip(path[:-1], path[1:]):
                if track_merged:
                    # keep track of the edges that were merged
                    merged_edges.append((u, v))

                # get edge between these nodes: 
                edge_data = G.get_edge_data(u, v)
                for attr in edge_data:
                    if attr in path_attributes:
                        # if this key already exists in the dict, append it to the
                        # value list
                        path_attributes[attr].append(edge_data[attr])
                    else:
                        # if this key doesn't already exist, set the value to a list
                        # containing the one value
                        path_attributes[attr] = [edge_data[attr]]

            # consolidate the path's edge segments' attribute values
            for attr in path_attributes:
                # merge list of list in path attr
                if attr in attrs_to_sum:
                    # if this attribute must be summed, sum it now
                    path_attributes[attr] = sum(path_attributes[attr])
                elif attr in attrs_to_max:
                    path_attributes[attr] = np.max(path_attributes[attr])
                elif attr in attrs_to_min:
                    path_attributes[attr] = np.min(path_attributes[attr])
                elif attr in attrs_to_unique:
                    path_attributes[attr] = np.unique(path_attributes[attr], axis=0)[0]
                # do not do the following processing since we have list of lists that does not work with set
                # elif len(set(path_attributes[attr])) == 1:
                #     # if there's only 1 unique value in this attribute list,
                #     # consolidate it to the single value (the zero-th):
                #     path_attributes[attr] = path_attributes[attr][0]
                # else:
                #     # otherwise, if there are multiple values, keep one of each
                #     path_attributes[attr] = list(set(path_attributes[attr]))

            # update link capacity that may no longer be valid
            path_attributes['capacity'] = compute_capacity(
                path_attributes['lane'], path_attributes['speed'],
                min_lane=path_attributes['capacity_param'][0],
                max_speed=path_attributes['capacity_param'][1]
            ) 

            # construct the new consolidated edge's geometry for this path
            # path_attributes["geometry"] = LineString(
            #     [Point((G.nodes[node]["x"], G.nodes[node]["y"])) for node in path]
            # )
            multi_line = MultiLineString(path_attributes["geometry"])
            path_attributes["geometry"] = linemerge(multi_line)

            if track_merged:
                # add the merged edges as a new attribute of the simplified edge
                path_attributes["merged_edges"] = merged_edges

            # add the nodes and edge to their lists for processing at the end
            all_nodes_to_remove.extend(path[1:-1])
            all_edges_to_add.append(
                {"origin": path[0], "destination": path[-1], "attr_dict": path_attributes}
            )

    # for each edge to add in the list we assembled, create a new edge between
    # the origin and destination
    for edge in all_edges_to_add:
        G.add_edge(edge["origin"], edge["destination"], **edge["attr_dict"])

    # finally remove all the interstitial nodes between the new edges
    G.remove_nodes_from(set(all_nodes_to_remove))

    # mark graph as having been simplified
    G.graph["simplified2"] = True

    return G


def compute_capacity(lane, speed, min_lane=2, max_speed=60):
    """compute link capacity based on lane and speed

    Args:
        lane (int): lane number
        speed (flow): speed limit on lane.
        min_lane (int): minimum number of lanes in one direction of a highway link. Defaults to 2
        max_speed (int, optional): max speed on a highway link. Defaults to 60.

    Returns:
        capacity (float): link flow capacity related to lane num. and speed limit
    """

    use_lane = lane if lane>=min_lane else min_lane

    capacity = use_lane/max_speed * speed
    
    return capacity


def generate_compute_graph(G):
    """Create a barebone graph model for nx computations

    Args:
        G (Graph or MultiGraph): must has been 2-d simplified and it can convert to Graph directly if needed
        min_lane (int, optional): minimum number of lanes for a highway link in one direction. Defaults to 2.
        max_speed (int, optional): max speed on a highway link. Defaults to 60.

    Returns:
        H (MultiDiGraph): Graph with added attributes
    """
    
    G_comp = nx.Graph()
    
    G_comp.add_nodes_from(G.nodes)

    for u, v, data in G.edges(data=True):
        lane = data['lane']
        speed = data['speed']
        capacity = data['capacity']
        bridge = data['bridge_id']
        length = data['length']
        G_comp.add_edge(u, v, lane=lane, speed=speed,
                        capacity=capacity, length=length, bridge_id=bridge)
    
    return G_comp


def identify_end_nodes(G, state_polygon=None, save_nodes=False):
    """Identify boundary nodes and all end nodes.

    Args:
        G (Graph, DiGraph, MultiGraph, MultiDiGraph): graph of highway network
        state_polygon (GeoDataFrame): State Polygon.
        save_nodes (bool, optional): whether save node GIS files. Defaults to False.

    Returns:
        bnd_nodes: list of nodes outside of state boundary
        end_nodes: list of all dead-end nodes
    """

    if isinstance(state_polygon, str):
        polygon = gpd.read_file(state_polygon)
    elif isinstance(state_polygon, gpd.GeoDataFrame):
        polygon = state_polygon
    else:
        sys.exit("Must provide valid state_polygon")
        
    # use reset_index to convert osmid from index to column
    nodes_gdf = utils_graph.graph_to_gdfs(G, edges=False).reset_index()

    # select nodes outside OR boundaries
    polygon.to_crs(crs=nodes_gdf.crs, inplace=True)
    bnd_nodes_gdf = nodes_gdf.overlay(polygon, how='difference')
    if save_nodes:
        bnd_nodes_gdf.to_file('./tmp_bnd_nodes.gpkg', drive='gpkg')
    bnd_nodes = bnd_nodes_gdf['osmid'].to_list()

    # all dead-end nodes
    spn = dict(nx.degree(nx.Graph(G)))
    end_nodes = [node for node, count in spn.items() if count <= 1]
    ends_gdf = nodes_gdf.loc[nodes_gdf['osmid'].isin(end_nodes)]
    if save_nodes:
        ends_gdf.to_file('./tmp_dead_ends.gpkg', drive='gpkg')
    
    return bnd_nodes, end_nodes


def generate_highway_graph(
    G_raw, crs='epsg:3857', consol_tolerance=100,
    min_lane=2, max_speed=60,
    save_graph=False, track_merged=False):

    """Generate computational and GIS graphs of highway network.

    Args:
        G_raw (MultiDiGraph): graph generated by osmnx
        crs (str, optional): CRS system, must be projected and use meter as unit. Defaults to 'epsg:3857'.
        consol_tolerance (int, optional): tolerance for node consolidation (in meter). Defaults to 100.
        min_lane (int, optional): minimum number of lanes for a highway link in one direction. Defaults to 2.
        max_speed (int, optional): max speed on a highway link. Defaults to 60.
        save_graph (bool, optional): whether to save intermediate graphs. Defaults to False.
        track_merged (bool, optional): wheterh to track merged links during 2-d simplification. Defaults to False.

    Returns:
        G_comp (Graph): computational graph with minimal info for network analysis
        G_gis (MultiDiGraph): graph with all info that can save to GIS using osmnx
    """
        
    # STEP 1: remove parallel links. To do this successfully, we need to first
    # convert the graph to undirected graph. This will make sure we'll also
    # remove one-way circles between two nodes. Then, we remove parallal links
    # by leaving only the shortest edge (this convert Graph back to DiGraph)
    G_raw = ox.project_graph(G_raw, to_crs=crs)
    G_undir = ox.get_undirected(G_raw)
    G_nopar = ox.get_digraph(G_undir, weight='length')
    # add a flag to know graph can be converted to undirected graph (and vice
    # versa) safely
    
    if save_graph:
        G_plot = nx.MultiDiGraph(G_nopar)
        ox.save_graph_geopackage(G_plot, filepath="./tmp_G_nopar.gpkg")
    

    # STEP 2: consolidate nodes to simply graph. Remove parallel links
    # afterwards (same as STEP 1 operations). Note the
    # consolidate_intersections() only works with MultiDiGraph or MultiGraph
    G_nopar = nx.MultiDiGraph(G_nopar)
    G_consol = ox.consolidate_intersections(
        G_nopar, tolerance=consol_tolerance,
        dead_ends=True, rebuild_graph=True
    )
    Gc_undir = ox.get_undirected(G_consol)
    Gc_nopar = ox.get_digraph(Gc_undir, weight='length')

    if save_graph:
        G_plot = nx.MultiDiGraph(Gc_nopar)
        ox.save_graph_geopackage(G_plot, filepath="./tmp_Gc_nopar.gpkg")
    

    # STEP 3: append computational attributes to links in preparation for
    # further simplification (based on 2-degree nodes). Note if Graph is
    # directed, the number of bridges may double (this might be needed, but
    # mostly not). Do not use ox.get_undirected since it expects a MultiDiGrpah
    Gc_nopar = nx.to_undirected(Gc_nopar)
    _append_compute_attributes(Gc_nopar, min_lane=min_lane, max_speed=max_speed,
                               inplace=True)


    # STEP 4: further simplify graph by removing 2-degree nodes that do no
    # connect to a bridge link
    Gc_gis = _simplify_graph_d2(Gc_nopar, track_merged=track_merged)
    if save_graph:
        G_plot = nx.MultiDiGraph(Gc_gis)
        ox.save_graph_geopackage(G_plot, filepath="./tmp_Gc_gis.gpkg")


    # STEP 5: generate simple graph for network computation
    Gc_comp = generate_compute_graph(Gc_gis)

    # convert to a MultiDiGraph so it can be saved as gpkg without issue
    Gc_gis = nx.MultiDiGraph(Gc_gis)

    return Gc_comp, Gc_gis


def generate_od_pairs(G, end_nodes=None, length='length',
                      min_distance=0., max_distance=1e12):
    """generate od pairs based on end nodes. They are generated based on:
    * each bridge has at least one OD
    * for one bridge, the OD is selected from all end-node pairs whose 
    shortest path uses that bridge. When multiple are available, select
    the shortest one
    * the final list is the unique OD pairs associated all bridges

    Args:
        G (Graph, DiGraph, MultiGraph, MultiDiGraph): highway network
        end_nodes (list): a list of end nodes to be considered as OD
        length (str, optional): _description_. Defaults to 'length'.
        min_distance (float, optional): minimum distance to be
        considered as an OD pair. This is to avoid super short OD paths.
        Defaults to 0.
        max_distance (float, optional): maximum distance for initializeing 
        OD distance associated with each bridge. Defaults to 1e12.

    Returns:
        unique_pairs (list): list of od pairs
        shortest_path_log (DataFrame): table including all OD pairs 
        associated with different bridges
    """

    assert end_nodes is not None, "Must provide a list of end_nodes"

    bridge_edges = [i for i in G.edges if G.get_edge_data(*i)['bridge_id']>0]
    all_od_pairs = itertools.permutations(end_nodes, 2)    # use permutations because order of the two nodes matters

    shortest_path_log = pd.DataFrame()
    shortest_path_log['bridge'] = bridge_edges
    shortest_path_log['shortest'] = max_distance
    shortest_path_log['od'] = [(0,0)]*len(bridge_edges)

    for o,d in all_od_pairs:
        paths = nx.all_shortest_paths(G, o, d, weight=length)
        for path in paths:
            path_graph = nx.path_graph(path)
            path_length = nx.path_weight(G, path_graph, weight=length)
            if path_length < min_distance:
                continue

            for bridge in bridge_edges:
                if path_graph.has_edge(*bridge):
                    current_shortest = shortest_path_log.loc[shortest_path_log.bridge==bridge, 'shortest'].values[0]
                    if current_shortest > path_length:
                        indx = shortest_path_log.index[shortest_path_log.bridge == bridge][0]
                        shortest_path_log.at[indx, 'shortest'] = np.minimum(current_shortest, path_length)
                        shortest_path_log.at[indx, 'od'] = (o,d)

    all_pairs = shortest_path_log['od'].to_list()

    unique_pairs = np.unique(all_pairs, axis=0)
    unique_pairs = unique_pairs[np.all(unique_pairs, axis=1)]

    return unique_pairs, shortest_path_log


def bridge_path_capacity(G, bridge_path: pd.DataFrame, capacity='capacity'):

    bridge_path['flow'] = -1
    bridge_path['bridge_id'] = -1
    path_capacity = bridge_path.copy()

    for index, df_row in path_capacity.iterrows():
        od = df_row.od
        bridge = df_row.bridge
        df_row['bridge_id'] = G.get_edge_data(*bridge)['bridge_id']

        flow = bridge_path[bridge_path.bridge==bridge].iloc[0].at['flow']
        if od[0] == od[1]:
            # special case when a bridge is not on any od path. Set flow=0 so
            # failure of that bridge will have no consequences
            df_row['flow'] = 0
        elif flow == -1:
            flow, _ = nx.maximum_flow(G, od[0], od[1], capacity=capacity)
            bridge_path.loc[bridge_path.od==od, 'flow'] = flow
            df_row['flow'] = flow
        else:
            df_row['flow'] = flow
        
        # update path_capacity
        path_capacity.loc[index] = df_row

    return path_capacity

# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # download/load and generate raw osmnx Graph
    # place = {"state": "Oregon", "country": "USA"}
    # filter = '["highway"~"motorway|trunk|primary"]'
    # G_raw = ox.graph_from_place(place, truncate_by_edge=True, custom_filter=filter)
    # ox.save_graph_geopackage(G_raw, filepath="./tmp/or_hwy2prim_raw.gpkg")
    G_raw = ox.load_graphml(filepath="./tmp/or_hwy2prim_raw.graphml")

    fig, ax = ox.plot_graph(G_raw)
    fig.savefig('./tmp/or_hwy2prim_consol100.png', dpi=600)

    G_comp, G_gis = generate_highway_graph(
        G_raw, crs='epsg:3857', consol_tolerance=100,
        min_lane=2, max_speed=60)

    nx.write_graphml(G_comp, './tmp/or_comp_graph.graphml')

    bnd_nodes, end_nodes = identify_end_nodes(
        G_gis, state_polygon='./gis/OR-boundary/OR-boundary.shp')

    bnd_od, shortest_path_log = generate_od_pairs(
        G_comp, end_nodes=bnd_nodes, min_distance=50e3)
    np.savez('./tmp/bnd_od.npz', bnd_od=bnd_od)
    shortest_path_log.to_pickle('./tmp/bridge_path_bnd.pkl')

    end_od, shortest_path_log2 = generate_od_pairs(
        G_comp, end_nodes=end_nodes, min_distance=50e3)
    np.savez('./tmp/end_od.npz', end_od=end_od)
    shortest_path_log2.to_pickle('./tmp/bridge_path_end.pkl')

    path_capacity = bridge_path_capacity(G_comp, shortest_path_log)

