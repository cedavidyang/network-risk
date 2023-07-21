# %%
import sys
import numpy as np
import pandas as pd
import networkx as nx
import itertools
import multiprocess as mp
import queue


def bridge_capacity(G, capacity='capacity'):

    bridge_id = nx.get_edge_attributes(G, 'bridge_id')
    max_id = max(bridge_id.values())

    capacity_array = []
    for key, id in bridge_id.items():
        if id > 0:
            capacity = G[key[0]][key[1]][capacity]
            capacity_array.append(capacity)
    
    capacity_array = np.array(capacity_array)

    n_bridge = len(capacity_array)
    assert n_bridge == max_id, "max bridge id must be the same as the bridge number"

    return capacity_array
    

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
    
    n_bridge = len(path_capacity)
    max_id = path_capacity['bridge_id'].max()
    assert n_bridge == max_id, "max bridge id must be the same as the bridge number"

    capacity_array = np.zeros(n_bridge)
    for i in range(n_bridge):
        capacity_array[i] = path_capacity[path_capacity['bridge_id']==(i+1)].iloc[0].at['flow']

    return capacity_array, path_capacity


def compute_add_risk(pf_array_smps, cost_array):

    agency_risk_smps = (pf_array_smps @ cost_array) / cost_array.sum()
    
    return agency_risk_smps


def net_capacity(G, od_pairs, capacity='capacity'):
    total_flow = 0
    for od in od_pairs:
        flow, flow_dict = nx.maximum_flow(G, od[0], od[1], capacity=capacity)
        total_flow += flow
    return total_flow


def damaged_net_capacity(G, od_pairs, damage_condition, b_key='bridge_id',
        remain_capacity=0.5, capacity='capacity'):
    G1 = G.copy()
    for u, v, data in G1.edges(data=True):
        if (data[b_key] != 0) and (damage_condition[data[b_key]-1]):
            original_capacity = data[capacity]
            data[capacity] = np.minimum(remain_capacity, original_capacity)
    total_flow = 0
    for od in od_pairs:
        flow, flow_dict = nx.maximum_flow(G1, od[0], od[1], capacity=capacity)
        total_flow += flow
    return total_flow


def flow_conseq(flow, min_flow=0., max_flow=1.):
    util = (max_flow-flow)/(max_flow-min_flow)
    return util


def identify_damage_condition(n_br, damage_netdb=None, pf_array=None,
                              include_scenario: int|str ='all', missing_value=-1):
    
    if include_scenario == 'all':
        total_scenario = 2**n_br - 1
    elif isinstance(include_scenario, (int, np.int64)):
        total_scenario = include_scenario
    else:
        sys.exit("include_scenario must be 'all' or an integer")

    if pf_array is not None:
        sorting_indx = pf_array.argsort()[::-1]
    else:
        sorting_indx = np.arange(n_br)

    all_condition_key = []
    nsc = 0
    for failed in range(1, n_br):
        for indx in itertools.combinations(sorting_indx, failed):
            condition_key = tuple(sorted(indx))
            all_condition_key.append(condition_key)

            nsc += 1
            if nsc >= total_scenario:
                break

        if nsc >= total_scenario:
            break

    # initialize the dictionary
    if damage_netdb is None:
        missing_condition_key = all_condition_key
    else:
        existing_key = [key for key, value in damage_netdb.items()
                            if (value != missing_value) and (key not in ('max', 'min'))]
        missing_condition_key = list(set(all_condition_key)-set(existing_key))


    return missing_condition_key


def generate_damage_netdb(
        G, od_pairs, pf_array=None, remain_capacity=0., damage_netdb=None, 
        include_scenario: int|str ='all', capacity='capacity',
        missing_value=-1, n_jobs=1, timeout=None) -> dict[tuple[int, ...]|str, float]:
    
    bridge_id = nx.get_edge_attributes(G, 'bridge_id')
    n_br = max(bridge_id.values())

    missing_condition = identify_damage_condition(
        n_br, damage_netdb=damage_netdb, pf_array=pf_array,
        include_scenario=include_scenario, missing_value=missing_value)

    if damage_netdb is None:
        # initialize the dictionary
        damage_netdb = dict()
    
    # parallel processing if asked
    def task_func(args):
        condition_key, q = args[0], args[1]
        damage_condition = np.zeros(n_br, dtype=bool)
        damage_condition[(condition_key,)] = True
        flow = damaged_net_capacity(G, od_pairs, damage_condition,
                                    remain_capacity=remain_capacity,
                                    capacity=capacity)
        q.put((condition_key, flow))

    if (n_jobs > 1) and (len(missing_condition)>0):
        with mp.Manager() as manager:
            mp_q = manager.Queue()
            with mp.Pool(n_jobs) as pool:
                try:
                    pool.map_async(
                        task_func, 
                        zip(missing_condition,itertools.repeat(mp_q)),
                    ).get(timeout)
                except mp.TimeoutError:
                    pass
            mp_q.put('STOP')
            for key, flow in iter(mp_q.get, 'STOP'):
                damage_netdb[key] = flow
    else:
        q: queue.Queue = queue.Queue()
        try:
            op = list(map(task_func, zip(missing_condition, itertools.repeat(q))))
        except KeyboardInterrupt:
            pass
        q.put('STOP')
        for key, flow in iter(q.get, 'STOP'):
            damage_netdb[key] = flow

    if include_scenario in ('all', 2**n_br-1):
        damage_netdb['max'] = 0
        damage_netdb['min'] = 0
    else:
        # assume the max achievable flow in the un-evaluated cases is the minimum flow in the evaluated cases 
        if ('min' not in damage_netdb.keys()) or (damage_netdb['min'] == missing_value):
            damage_condition = np.ones(n_br, dtype=bool)
            flow = damaged_net_capacity(G, od_pairs, damage_condition,
                                        remain_capacity=remain_capacity, capacity=capacity)
            damage_netdb['min'] = flow

        damage_netdb['max'] = min([v for k,v in damage_netdb.items() if k not in ('max', 'min')])

    return damage_netdb


def expected_maxflow_drop(
        G, od_pairs, pf_array=None, sort_pf=False, remain_capacity=0,
        max_flow=-1, damage_netdb=None, include_scenario: int|str ='all',
        capacity='capacity', n_jobs=1, missing_value=-1, timeout=None):

    if sort_pf:
        assert pf_array is not None, "must provide pf_array if sort_pf=True"
        pf4db = pf_array
    else:
        pf4db = None

    n_br = len(pf_array)
    if (include_scenario == 'all') or isinstance(include_scenario, (int, np.int64)):
        condition_keys = identify_damage_condition(
            n_br, damage_netdb=None, pf_array=pf4db,
            include_scenario=include_scenario, missing_value=missing_value)
        damage_netdb = generate_damage_netdb(
            G, pf_array=pf4db, damage_netdb=damage_netdb, od_pairs=od_pairs,
            remain_capacity=remain_capacity, include_scenario=include_scenario,
            capacity=capacity, n_jobs=n_jobs, missing_value=missing_value,
            timeout=timeout)
    elif include_scenario == 'custom':
        assert damage_netdb is not None, "must provide damage_netdb when include_scenario='custom'"
        condition_keys = [k for k in damage_netdb.keys() if k not in ('max', 'min')]
    else:
        sys.exit("include_scenario must be 'all' or an integer or 'custom'")
    
    total_scenario = len(condition_keys)
    scenario_probs = np.zeros(total_scenario)
    scenario_util = np.zeros(total_scenario)
    for nsc, key in enumerate(condition_keys):
        damage_condition = np.zeros(n_br, dtype=bool)
        damage_condition[(key,)] = True
        prob = np.exp(np.log(pf_array[damage_condition]).sum() +
                      np.log(1-pf_array[np.logical_not(damage_condition)]).sum())

        if key in damage_netdb.keys():
            flow = damage_netdb[key]
        else:
            flow = 0.5*(damage_netdb['max']+damage_netdb['min'])

        scenario_probs[nsc] = prob
        scenario_util[nsc] = flow

    scenario_util = flow_conseq(scenario_util, min_flow=0, max_flow=max_flow)

    if total_scenario == 2**n_br - 1:
        mean_net_drop = scenario_probs @ scenario_util
    else:
        # assume the max achievable flow in the un-evaluated cases is the minimum flow in the evaluated cases 
        p_all_survive = np.exp(np.log(1-pf_array).sum())
        remain_prob = 1 - np.sum(scenario_probs) - p_all_survive
        lb_conseq = flow_conseq(damage_netdb['max'], min_flow=0, max_flow=max_flow)
        ub_conseq = flow_conseq(damage_netdb['min'], min_flow=0, max_flow=max_flow)
        avg_remain_util = 0.5*(lb_conseq+ub_conseq)
        probs = np.hstack([scenario_probs, remain_prob])
        utils = np.hstack([scenario_util, avg_remain_util])

        mean_net_drop = probs @ utils

    return mean_net_drop, damage_netdb


def compute_net_risk(
        G, od_pairs, pf_array_smps, sort_pf=False, remain_capacity=0, max_flow=None,
        damage_netdb=None, include_scenario: int|str ='all',
        capacity='capacity', n_jobs=1, missing_value=-1, timeout=None):

    if max_flow is None:
        max_flow = net_capacity(G, od_pairs=od_pairs, capacity=capacity)

    nsmp, _ = pf_array_smps.shape

    net_risk_smps = np.ones(nsmp)*missing_value
    try:
        for i, pf_array in enumerate(pf_array_smps):
            net_risk, damage_netdb = expected_maxflow_drop(
                G, od_pairs, pf_array=pf_array, sort_pf=sort_pf,
                remain_capacity=remain_capacity, max_flow=max_flow,
                damage_netdb=damage_netdb, include_scenario=include_scenario,
                capacity=capacity, n_jobs=n_jobs,
                missing_value=missing_value, timeout=timeout)
            net_risk_smps[i] = net_risk
    except KeyboardInterrupt:
        pass

    return net_risk_smps, damage_netdb


def _damage_netdb_test():
    include_scenario = 2
    remain_capacity = 1/60*20

    od_data = np.load('./tmp/bnd_od.npz')
    od_pairs = od_data['bnd_od']
    G_comp = nx.read_graphml('./tmp/or_comp_graph.graphml', node_type=int)

    damage_netdb = generate_damage_netdb(
        G_comp, damage_netdb=None, od_pairs=od_pairs, remain_capacity=remain_capacity,
        include_scenario=include_scenario, capacity='capacity',
        n_jobs=1, missing_value=-1)

    damage_netdb0 = damage_netdb.copy()
    include_scenario = 8
    damage_netdb = generate_damage_netdb(
        G_comp, damage_netdb=damage_netdb0, od_pairs=od_pairs,
        remain_capacity=remain_capacity,
        include_scenario=include_scenario, capacity='capacity',
        n_jobs=4, missing_value=-1)


def _risk_compare_test():
    import os
    import pickle

    from datetime import datetime
    from scipy import stats
    from math import comb
    
    G_comp = nx.read_graphml('./tmp/or_comp_graph.graphml', node_type=int)
    bridge_path = pd.read_pickle('./tmp/bridge_path_bnd.pkl')
    # bridge_path = pd.read_pickle('./tmp/bridge_path_end.pkl')

    # use the following code to generate path_capacity
    path_capacity, path_capacity_df = bridge_path_capacity(
        G_comp, bridge_path)

    timeout0, timeout1, n_jobs = 4*60*60, 60, 80
    nsmp, n_br = 10, len(path_capacity)
    n_fail = 2
    fail_seed = 1
    remain_capacity = 1/60*20
    min_beta, max_beta = 0, 3

    beta_array_smps = stats.uniform.rvs(loc=min_beta, scale=max_beta-min_beta,
                                        size=(nsmp, n_br), random_state=fail_seed)
    pf_array_smps = stats.norm.cdf(-beta_array_smps)

    # ===============================================================
    # Additive risk
    # ===============================================================
    cost_array = path_capacity
    add_risk_smps = compute_add_risk(pf_array_smps, cost_array)

    # ===============================================================
    # Net risk
    # ===============================================================
    n_fail = np.minimum(n_br, n_fail)
    include_scenario = np.sum([comb(n_br, j) for j in range(1, n_fail+1)])


    # # explicitly obtain damage_netdb to allow keyboard interrupt
    od_data = np.load('./tmp/bnd_od.npz')
    od_pairs = od_data['bnd_od']
    max_flow = net_capacity(G_comp, od_pairs=od_pairs, capacity='capacity')

    # explicitly obtain damage_netdb to allow keyboard interrupt
    with open('./tmp/tmp_2023-07-18_14_20/damage_netdb.pkl', 'rb') as f_read:
        damage_netdb = pickle.load(f_read)
    # min_scenario = n_br
    min_scenario = include_scenario
    damage_netdb = generate_damage_netdb(
        G_comp, pf_array=None, damage_netdb=damage_netdb, od_pairs=od_pairs,
        remain_capacity=remain_capacity, include_scenario=min_scenario,
        capacity='capacity', n_jobs=n_jobs, missing_value=-1,
        timeout=timeout0)

    net_risk_smps, damage_netdb = compute_net_risk(
        G_comp, od_pairs, pf_array_smps, sort_pf=True,
        remain_capacity=remain_capacity, max_flow=max_flow,
        damage_netdb=damage_netdb, include_scenario=include_scenario,
        capacity='capacity', n_jobs=n_jobs, missing_value=-1,
        timeout=timeout1)
    
    # ===============================================================
    # Save results
    # ===============================================================
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H_%M')
    data_dir = os.path.join('./tmp', f'tmp_{time_stamp}')
    os.makedirs(data_dir, exist_ok=False)

    with open(os.path.join(data_dir, 'damage_netdb.pkl'), 'wb') as f:
        pickle.dump(damage_netdb, f)

    path_capacity_df.to_pickle(
        os.path.join(data_dir, 'path_capacity_df.pkl'))

    np.savez(os.path.join(data_dir, 'MC_smps.npz'),
             max_flow=max_flow,
             include_scenario=include_scenario,
             pf_array_smps=pf_array_smps,
             add_risk_smps=add_risk_smps,
             net_risk_smps=net_risk_smps)


if __name__ == '__main__':
    import os
    import pickle

    from datetime import datetime
    from scipy import stats
    from math import comb

    # ===============================================================
    # Use existing damage_netdb
    # ===============================================================
    
    G_comp = nx.read_graphml('./tmp/or_comp_graph.graphml', node_type=int)
    bridge_path = pd.read_pickle('./tmp/bridge_path_bnd.pkl')
    # bridge_path = pd.read_pickle('./tmp/bridge_path_end.pkl')

    # use the following code to generate path_capacity
    # path_capacity, path_capacity_df = bridge_path_capacity(
    #     G_comp, bridge_path)
    path_capacity_df = pd.read_pickle('./tmp/path_capacity_df.pkl') 
    path_capacity = path_capacity_df.sort_values('bridge_id')['flow'].to_numpy()

    nsmp, n_br = 100, len(path_capacity)
    n_fail = 2
    fail_seed = 1
    remain_capacity = 1/60*20
    min_beta, max_beta = 3, 4

    beta_array_smps = stats.uniform.rvs(loc=min_beta, scale=max_beta-min_beta,
                                        size=(nsmp, n_br), random_state=fail_seed)
    pf_array_smps = stats.norm.cdf(-beta_array_smps)

    # ===============================================================
    # Additive risk
    # ===============================================================
    cost_array = path_capacity
    add_risk_smps = compute_add_risk(pf_array_smps, cost_array)

    # ===============================================================
    # Net risk
    # ===============================================================

    od_data = np.load('./tmp/bnd_od.npz')
    od_pairs = od_data['bnd_od']
    max_flow = net_capacity(G_comp, od_pairs=od_pairs, capacity='capacity')

    # explicitly obtain damage_netdb to allow keyboard interrupt
    with open('./tmp/tmp_2023-07-17_13_41/damage_netdb.pkl', 'rb') as f_read:
        damage_netdb = pickle.load(f_read)
    
    include_scenario = 'custom'
    net_risk_smps, damage_netdb = compute_net_risk(
        G_comp, od_pairs, pf_array_smps, sort_pf=True,
        remain_capacity=remain_capacity, max_flow=max_flow,
        damage_netdb=damage_netdb, include_scenario=include_scenario,
        capacity='capacity', n_jobs=1, missing_value=-1,
        timeout=None)
    
    # ===============================================================
    # Save results
    # ===============================================================
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H_%M')
    data_dir = os.path.join('./tmp', f'tmp_{time_stamp}')
    os.makedirs(data_dir, exist_ok=False)

    path_capacity_df.to_pickle(
        os.path.join(data_dir, 'path_capacity_df.pkl'))

    np.savez(os.path.join(data_dir, 'MC_smps.npz'),
             max_flow=max_flow,
             include_scenario=include_scenario,
             pf_array_smps=pf_array_smps,
             add_risk_smps=add_risk_smps,
             net_risk_smps=net_risk_smps)

    # ===============================================================
    # try comparing bridge consequence from damage_netdb
    # ===============================================================
    bridge_cost = (max_flow-np.array([damage_netdb[(i,)] for i in range(n_br)]))/max_flow
    add_risk_smps2 = compute_add_risk(pf_array_smps, bridge_cost)