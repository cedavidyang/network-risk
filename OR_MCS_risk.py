# Crude Monte Carlo simulation of the Oregon highway network risk

import os
import pickle
import numpy as np
import networkx as nx
import osmnx as ox
from multiprocessing import Pool, cpu_count
import time
from datetime import datetime

from UQpy.distributions import Normal, Uniform, MultivariateNormal

from netrisk_TMCMC.net_util import get_damage_state, net_capacity, scenario_cost


# Parameters
max_analysis = 44123  # Equal with the number of unique system states of TMCMC simulation
max_smp = max_analysis
remain_capacity = 1/60*20
min_beta, max_beta = 0, 3

data_dir = './assets'
res_dir = './results'

seed_MCS = 1
seed_beta = 1


# Load computational graph and change graph to simple graph
edge_attr_dict = {
    'length': float, 'capacity': float, 'lane': int,
    'speed': float, 'bridge_id': int
}
G_multi = ox.load_graphml(
    os.path.join(data_dir, 'or_hw_comp.graphml'), 
    edge_dtypes=edge_attr_dict
)
G_comp = nx.DiGraph(G_multi)


# Load OD pairs
od_data = np.load(os.path.join(data_dir, 'bnd_od.npz'))
od_pairs = od_data['bnd_od']


# Get list of bridges and maximum flow
bridge_list = [b for _, _, b in G_comp.edges.data('bridge_id') if b > 0]
n_br = len(bridge_list)
max_flow = net_capacity(G_comp, od_pairs=od_pairs, capacity='capacity')


# Generate beta and condition samples
beta_array = Uniform(loc=min_beta, scale=max_beta-min_beta).rvs(nsamples=n_br, random_state=seed_beta).flatten()
pf_array = Normal().cdf(-beta_array)
condition_rv = MultivariateNormal(mean=[0.0]*n_br, cov=1.0)
condition_rvs_max = condition_rv.rvs(nsamples=max_smp, random_state=seed_MCS)

damage_condition = np.array([get_damage_state(condition_rvs_max[i].reshape((1, -1)), beta_array) for i in range(max_smp)])
unique_damage_condition, unique_indices, unique_counts = np.unique(damage_condition, return_index=True, return_counts=True, axis=0)


# Function for parallel computation
def worker_task(i):
    try:
        condition_rvs = condition_rvs_max[i].reshape((1, -1))
        damage_condition = get_damage_state(condition_rvs, beta_array)
        
        cost = scenario_cost(damage_condition=damage_condition, G=G_comp, od_pairs=od_pairs, 
                             remain_capacity=remain_capacity, capacity='capacity', max_flow=max_flow)
        return cost
    except KeyboardInterrupt:
        return None

if __name__ == "__main__":
    time0 = time.time()
    print("Start Crude Monte Carlo simulation")

    # Optimize the number of processes to the number of CPU cores
    pool_size = min(cpu_count(), len(condition_rvs_max))
    with Pool(pool_size) as pool:
        C_smp_list = pool.map(worker_task, range(len(condition_rvs_max)))

    # Remove None values in case of KeyboardInterrupt
    C_smp_list = [cost for cost in C_smp_list if cost is not None]

    time1 = time.time()
    C_smp_array = np.array(C_smp_list)
    MC_risk = np.mean(C_smp_array)
    MC_time = time1 - time0

    # Save results
    time_stamp = datetime.now().strftime('Y-%m-%d_%H_%M')
    res_dir = os.path.join(res_dir, f'MCS_run_{time_stamp}')
    os.makedirs(res_dir, exist_ok= False)

    np.savez(
        os.path.join(res_dir, 'results.npz'),
        cost=C_smp_array,
        MC_time=MC_time,
        evidence = MC_risk
    )

    print(f"Sampling Completed in: {MC_time} seconds")
    print(f"MCS Evidence estimated: {MC_risk}")
