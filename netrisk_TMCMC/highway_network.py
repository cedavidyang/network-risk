# %%
import os
import sys
import itertools
import networkx as nx
import osmnx as ox
import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.stats as stats

from osmnx import utils_graph
from shapely.ops import linemerge
from shapely.geometry import MultiLineString

_MIN_LANE, _MAX_SPEED = 2, 60.0
_CRS = 'epsg:3857'
_WEIGHT = 'length'

def _append_compute_attributes(
    G, min_lane=_MIN_LANE, max_speed=_MAX_SPEED,
):
    """Add numertical attributes needed for graph flow computation and further graph simplification

    Args:
        G (Graph, MultiDiGraph): original graph after preliminary simplification and consolidation
        min_lane (int, optional): minimum number of lanes for a highway link in one direction. Defaults to 2.
        max_speed (int, optional): max speed on a highway link. Defaults to 60.
        inplace (bool, optional): whether to add attributes in place. Defaults to True

    Returns:
        H (DiGraph): Graph with added attributes
    """
    
    if isinstance(G, (nx.Graph, nx.DiGraph)):
        H = G.copy()
    else:
        sys.exit("G must be nx objects")

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

        # Use the following assumption: capacity = lane/max_speed*speed
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
        
    return H


def _append_compute_attributes_from_file(
    G, beta_threshold=10,
    min_lane=_MIN_LANE, max_speed=_MAX_SPEED, 
    csv_file=None,
):
    """Use excel file to add numertical attributes needed for graph flow computation and further graph simplification

    Args:
        G (Graph, DiGraph): original graph after preliminary simplification and consolidation
        min_lane (int, optional): minimum number of lanes for a highway link in one direction. Defaults to 2.
        max_speed (int, optional): max speed on a highway link. Defaults to 60.
        inplace (bool, optional): whether to add attributes in place. Defaults to True
        weight (str, optional): edge weight attribute. Defaults to 'length'.
        csv_file (str, optional): csv file containing bridge failure probability (must have columns 'osmid' and 'failure_prob').

    Returns:
        H (DiGraph): Graph with added attributes
    """

    assert csv_file is not None, \
        "Must path to an csv file with 'osmid' and 'failure_prob' columns"


    # Retain only required columns 
    bridge_data = pd.read_csv(csv_file, usecols=['osmid', 'failure_prob'])

    # Calculate the link beta values (indepedent bridge failure)
    link_pf = 1 - (1 - bridge_data['failure_prob']).groupby(bridge_data['osmid']).prod()
    link_data = pd.DataFrame({'osmid': link_pf.index, 'link_pf': link_pf.values})   

    # Change the osmid's to string array
    link_data['osmid'] = link_data['osmid'].astype(str)

    # link_beta is the beta value for the link
    link_data['link_beta'] = -stats.norm.ppf(link_data['link_pf'])

    H = _append_compute_attributes(G, min_lane=min_lane, max_speed=max_speed)

    bridge_indx = 0
    for u, v, data in H.edges(data=True):

        osmid = data['osmid']
        osmid = osmid if isinstance(osmid, str) else str(osmid)
        if osmid in link_data['osmid'].values:
            beta = link_data.loc[link_data['osmid'] == osmid, 'link_beta'].values[0]

            # only include bridge links with small beta values
            if beta <= beta_threshold:
                fail = True
            else:
                fail = False
                beta = None

        else:
            fail = False
            beta = None

        if fail and data['birdge_id'] != 0:
            bridge_indx += 1
            bridge_id = bridge_indx
        else:
            bridge_id = 0
        
        H.edges[u,v].update({
            'bridge_id': bridge_id,
            'beta': beta
        })
        
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

            # update link capacity that may no longer be valid
            path_attributes['capacity'] = compute_capacity(
                path_attributes['lane'], path_attributes['speed'],
                min_lane=path_attributes['capacity_param'][0],
                max_speed=path_attributes['capacity_param'][1]
            ) 

            # construct the new consolidated edge's geometry for this path
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

    # STEP 3: For each edge to add in the list we assembled, create a new edge between
    # the origin and destination
    for edge in all_edges_to_add:
        G.add_edge(edge["origin"], edge["destination"], **edge["attr_dict"])

    # STEP 4: Finally remove all the interstitial nodes between the new edges
    G.remove_nodes_from(set(all_nodes_to_remove))

    # mark graph as having been simplified
    G.graph["simplified2"] = True

    return G


def compute_capacity(lane, speed, min_lane=_MIN_LANE, max_speed=_MAX_SPEED):
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


def generate_highway_graph(
    G_raw, crs=_CRS,
    min_lane=_MIN_LANE, max_speed=_MAX_SPEED,
    multi_weight=_WEIGHT,
    from_pf_file=None,
    consol_tolerance=0,
    save_tmp_graph=False,
    track_merged=False,
    save_to=None
):

    """Generate computational and GIS graphs of highway network.

    Args:
        G_raw (MultiDiGraph): graph generated by osmnx
        crs (str, optional): CRS system, must be projected and use meter as unit. Defaults to 'epsg:3857'.
        min_lane (int, optional): minimum number of lanes for a highway link in one direction. Defaults to 2.
        max_speed (int, optional): max speed on a highway link. Defaults to 60.
        from_pf_file (str, optional): path to a csv file containing bridge failure probability. Defaults to None.
        multi_weight (str, optional): edge weight attribute to convert multigraph to simple graph. Defaults to 'length'.
        consol_tolerance (int, optional): tolerance for node consolidation (in meter). Defaults to 100.
        save_graph (bool, optional): whether to save intermediate graphs. Defaults to False.
        track_merged (bool, optional): wheterh to track merged links during 2-d simplification. Defaults to False.

    Returns:
        G_comp (Graph): computational graph with minimal info for network analysis
        G_gis (MultiDiGraph): graph with all info that can save to GIS using osmnx
    """

    if save_tmp_graph:
        # create a tmp folder if not exist
        os.makedirs("./tmp", exist_ok=True)
    
    # STEP 1: consolidate nodes to simply graph.   
    # TODO: check if code works fine when consol_tolerance>0
    G_raw = ox.project_graph(G_raw, to_crs=crs)
    if consol_tolerance > 0:
        Gc_raw = ox.consolidate_intersections(
            G_raw, tolerance=consol_tolerance,
            dead_ends=True, rebuild_graph=True
        )
    else:
        Gc_raw = G_raw

    # STEP 2: remove parallel links. By default, we remove parallal links
    # by leaving only the shortest edge (this convert Graph back to DiGraph)
    Gc_nopar = ox.get_digraph(Gc_raw, weight=multi_weight)
    if save_tmp_graph:
        nx.write_graphml(Gc_nopar, "./tmp/Gc_nopar.graphml")
        G_plot = nx.MultiDiGraph(Gc_nopar)
        ox.save_graph_geopackage(G_plot, filepath="./tmp/Gc_nopar.gpkg")

    # STEP 3: append computational attributes to links in preparation for
    # further simplification (based on 2-degree nodes). Note if Graph is
    # directed, the number of bridges may double (this might be needed, but
    # mostly not). Do not use ox.get_undirected since it expects a MultiDiGrpah
    if from_pf_file is None:
        Gc_nopar = _append_compute_attributes(
            Gc_nopar, min_lane=min_lane, max_speed=max_speed,
        )
    else:
        Gc_nopar = _append_compute_attributes_from_file(
            Gc_nopar, min_lane=min_lane, max_speed=max_speed,
            csv_file=from_pf_file,
        )

    # STEP 4: further simplify graph by removing 2-degree nodes that do no
    # connect to a bridge link
    Gc_gis = _simplify_graph_d2(Gc_nopar, track_merged=track_merged)
    if save_tmp_graph:
        nx.write_graphml(Gc_gis, "./tmp/Gc_gis.graphml")
        G_plot = nx.MultiDiGraph(Gc_gis)
        ox.save_graph_geopackage(G_plot, filepath="./tmp/Gc_gis.gpkg")

    if save_to is not None:
        nx.write_graphml(Gc_gis, save_to+".graphml")
        G_plot = nx.MultiDiGraph(Gc_gis)
        ox.save_graph_geopackage(G_plot, filepath=save_to+".gpkg")

    return Gc_gis


def generate_compute_graph(G, save_to=None):
    """Create a barebone graph model for nx computations

    Args:
        G (nx graph): graph generated by generate_highway_graph
    Returns:
        G_comp (Graph): Graph with added attributes
    """

    if G.is_directed():
        G_comp = nx.DiGraph()
    else:
        G_comp = nx.Graph()
    
    G_comp.add_nodes_from(G.nodes)

    for u, v, data in G.edges(data=True):
        lane = data['lane']
        speed = data['speed']
        capacity = data['capacity']
        bridge = data['bridge_id']
        length = data['length']
        beta = data['beta'] if 'beta' in data else None
        G_comp.add_edge(
            u, v, lane=lane, speed=speed,
            capacity=capacity, length=length,
            bridge_id=bridge, beta=beta
        )

    if save_to is not None:
        nx.write_graphml(G_comp, save_to+".graphml")
    
    return G_comp


def identify_bnd_nodes(G, state_polygon=None, save_nodes=False):
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


def generate_od_pairs(
    G, end_nodes=None, length=_WEIGHT,
    min_distance=0., max_distance=1e12,
    save_to=None,
):
    """generate od pairs based on end nodes. They are generated based on:
    * each bridge has at least one OD
    * for one bridge, the OD is selected from all end-node pairs whose 
    shortest path uses that bridge. When multiple are available, select
    the shortest one
    * the final list is the unique OD pairs associated all bridges

    Args:
        G (Graph, DiGraph, MultiGraph, MultiDiGraph): highway network
        end_nodes (list): a list of end nodes to be considered as OD
        length (str, optional): key name to compute path length.
        Defaults to 'length'.
        min_distance (float, optional): minimum distance to be
        considered as an OD pair. This is to avoid super short OD paths.
        Defaults to 0.
        max_distance (float, optional): maximum distance for initializeing 
        OD distance associated with each bridge. Defaults to 1e12.
        save_to (str, optional): path to save the OD pairs. Defaults to
        None (no saving).

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
        # check if d is reachable from o
        if not nx.has_path(G, o, d):
            continue

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

    if save_to is not None:
        # save the OD pairs to assets
        np.savez(save_to+'.npz', bnd_od=unique_pairs)
        list_unique_nodes = np.unique(unique_pairs)
        subgraph = G.subgraph(list_unique_nodes)
        unique_nodes_gdf = utils_graph.graph_to_gdfs(subgraph, nodes=True, edges=False)
        unique_nodes_gdf.to_file(save_to, driver='GPKG')

    return unique_pairs, shortest_path_log


def bridge_path_capacity(G, bridge_path: pd.DataFrame, capacity='capacity'):
    """For each bridge, get the max flow along the shortest path using that bridge.

    Args:
        G (Graph): computational graph representing the highway
        bridge_path (pd.DataFrame): pandas DataFrame containing the bridge link,
        the shortest path using that link, and the OD of that path. This is
        obtained from `generate_OD_pairs`.
        capacity (str, optional): key name used to compute maximum flow.
        Defaults to 'capacity'.

    Returns:
        pd.DataFrame: similar to bridge_path but include bridge id and the maximum
        flow on the shortest path usnig the bridge
    """

    path_capacity = bridge_path.copy()
    path_capacity['flow'] = -1
    path_capacity['bridge_id'] = -1

    for index, df_row in path_capacity.iterrows():
        od = df_row.od
        bridge = df_row.bridge
        df_row['bridge_id'] = G.get_edge_data(*bridge)['bridge_id']

        flow = path_capacity[path_capacity.bridge==bridge].iloc[0].at['flow']
        if od[0] == od[1]:
            # special case when a bridge is not on any od path. Set flow=0 so
            # failure of that bridge will have no consequences
            df_row['flow'] = 0
        elif flow == -1:
            flow, _ = nx.maximum_flow(G, od[0], od[1], capacity=capacity)
            # bridge_path.loc[bridge_path.od==od, 'flow'] = flow
            df_row['flow'] = flow
        
        # update path_capacity
        path_capacity.loc[index] = df_row

    return path_capacity


if __name__ == '__main__':
    pass
