# %%
import numpy as np
import networkx as nx
from typing import Optional

from UQpy.distributions import Normal
from .type_verifier import Numpy2DBooleanArray, Numpy2DFloatArray, NumpyFloatArray, Numpy2DIntArray


def net_capacity(G, od_pairs, capacity='capacity'):
    total_flow = 0
    for od in od_pairs:
        flow, flow_dict = nx.maximum_flow(G, od[0], od[1], capacity=capacity)
        total_flow += flow
    return total_flow


def get_damage_state(condition_var, beta_array):

    damage_condition = condition_var <= -beta_array

    return  damage_condition


def damaged_net_capacity(G, od_pairs, damage_condition, b_key='bridge_id',
        remain_capacity=0.5, capacity='capacity'):
    G1 = G.copy()
    for u, v, data in G1.edges(data=True):
        if (data[b_key] != 0) and (damage_condition[data[b_key]-1]):
            original_capacity = data[capacity]
            data[capacity] = np.minimum(remain_capacity, original_capacity)
    total_flow = 0
    for od in od_pairs:
        flow = nx.maximum_flow_value(G1, od[0], od[1], capacity=capacity)
        total_flow += flow
    return total_flow


def scenario_logp(condition_var, beta_array, from_condition=False):

    if from_condition:
        damage_condition = condition_var.astype(bool)
    else:
        damage_condition = get_damage_state(condition_var, beta_array)
    pf_array = Normal().cdf(-beta_array)

    n_smps = condition_var.shape[0]
    logp_sum = np.zeros(n_smps)
    for i, condition in enumerate(damage_condition):
        logp_fail = np.log(pf_array[condition])
        logp_safe = np.log(1-pf_array[np.logical_not(condition)])
        logp_sum[i] = np.sum(logp_fail) + np.sum(logp_safe)
    
    return logp_sum


def scenario_cost(
        damage_condition: Numpy2DBooleanArray,
        G: Optional[nx.Graph] = None,
        od_pairs: Optional[list[tuple[int,int]]|Numpy2DIntArray] = None,
        remain_capacity: float = 0.,
        capacity: str = 'capacity',
        max_flow: float = 1.,
        damage_db: None|dict[tuple[int,...], float] = None,
        epsilon=1e-6,
    ) -> NumpyFloatArray:

    if damage_db is None:
        damage_db = dict()

    n_smps = damage_condition.shape[0]
    cost = np.zeros(n_smps)
    for i, condition in enumerate(damage_condition):
        key = tuple(np.where(condition)[0])
        # if key in damage_db.keys():   # this is extremely slow, will result in runtimeerror, when one process is editing the dict in the meantime
        if key in damage_db:
            flow = damage_db[key]
        else:
            flow = damaged_net_capacity(G, od_pairs, condition,
                                        remain_capacity=remain_capacity,
                                        capacity=capacity)
            damage_db[key] = flow

        cost[i] = (max_flow-flow)/max_flow + epsilon

    return cost


def scenario_logC(
        condition_var: Numpy2DFloatArray,
        beta_array: NumpyFloatArray,
        from_condition: bool =False,
        G: Optional[nx.Graph] = None,
        od_pairs: Optional[list[tuple[int,int]]|Numpy2DIntArray] = None,
        remain_capacity: float = 0.,
        capacity: str = 'capacity',
        max_flow: float = 1.,
        damage_db: None|dict[tuple[int,...], float] =None,
        epsilon=1e-6,
    ):

    if from_condition:
        damage_condition = condition_var.astype(bool)
    else:
        damage_condition = get_damage_state(condition_var, beta_array)
    
    cost = scenario_cost(
        damage_condition, G=G, od_pairs=od_pairs, remain_capacity=remain_capacity,
        capacity=capacity, max_flow=max_flow, damage_db=damage_db, epsilon=epsilon,
    )
    
    logC = np.log(cost)

    return logC,


if __name__ == '__main__':
    pass