# %%
import numpy as np
import networkx as nx
from UQpy.distributions import Normal
from typing import Optional

from type_verifier import Numpy2DBooleanArray, Numpy2DFloatArray, NumpyFloatArray, Numpy2DIntArray
from highway_risk import damaged_net_capacity


def get_damage_state(condition_var, beta_array):

    damage_condition = condition_var <= -beta_array

    return  damage_condition


def scenario_logp(condition_var, beta_array, from_condition=False):

    if from_condition:
        damage_condition = condition_var.astype(bool)
    else:
        damage_condition = get_damage_state(condition_var, beta_array=beta_array)
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
        damage_db: None|dict[tuple[int,...], float] =None,
        epsilon=1e-6,
    ) -> NumpyFloatArray:

    # TODO: potential parallelization can happen here
    if damage_db is None:
        damage_db = dict()

    n_smps = damage_condition.shape[0]
    cost = np.zeros(n_smps)
    for i, condition in enumerate(damage_condition):
        key = tuple(np.where(condition)[0])
        if key in damage_db.keys():
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
        damage_condition = get_damage_state(condition_var, beta_array=beta_array)
    
    cost = scenario_cost(
        damage_condition, G=G, od_pairs=od_pairs, remain_capacity=remain_capacity,
        capacity=capacity, max_flow=max_flow, damage_db=damage_db, epsilon=epsilon
    )
    
    logC = np.log(cost)

    return logC,


if __name__ == '__main__':
    import os
    import pickle
    import pandas as pd
    import numpy as np
    import itertools
    import multiprocess as mp

    from UQpy.sampling import MetropolisHastings
    from UQpy.distributions import Normal, Uniform, MultivariateNormal

    from parallel_tempMCMC import SequentialTemperingMCMCpar
    from highway_risk import bridge_path_capacity, net_capacity

    _BURN_LENGTH, _JUMP, _N_CHAINS = 10000, 100, 10
    n_jobs = 8
    remain_capacity = 1/60*20
    min_beta, max_beta, n_smp = 0, 3, 10000
    n_br = 5
    max_flow = 100.0    # assumed max_flow so all tested bridges will have a non-zeros cost
    cov = 1.0

    # test with the first 5 bridges
    G_comp = nx.read_graphml('./tmp/or_comp_graph.graphml', node_type=int)

    for u, v, b in G_comp.edges.data('bridge_id'):
        if b > n_br:
            G_comp[u][v]['bridge_id'] = 0
    bridge_list = [b for _, _, b in G_comp.edges.data('bridge_id') if b>0]
    assert len(bridge_list) == max(bridge_list), "maximum bridge must equal to num. of bridges"
    od_data = np.load('./tmp/bnd_od.npz')
    od_pairs = od_data['bnd_od']
    with open('./tmp/tmp_2023-07-20_05_47/damage_netdb.pkl', 'rb') as f_read:
        damage_netdb = pickle.load(f_read)
    damage_db = dict()
    # for n in range(1, n_br+1):
    #     for key in itertools.combinations(range(1, n_br+1), n):
    #         if key in damage_netdb.keys():
    #             damage_db[key] = damage_netdb[key]
    for n in range(1, n_br+1):
        if (n,) in damage_netdb.keys():
            damage_db[(n,)] = damage_netdb[(n,)]


    beta_array = Uniform(loc=min_beta, scale=max_beta-min_beta,).rvs(
        nsamples=n_br, random_state=1)
    beta_array = beta_array.flatten()
    pf_array = Normal().cdf(-beta_array)

    # define intermediate function
    def _log_pdf_intermediate(
            x, b,
            beta_array: NumpyFloatArray = beta_array,
            from_condition: bool =False,
            G: Optional[nx.Graph] = G_comp,
            od_pairs: Optional[list[tuple[int,int]]] = od_pairs,
            remain_capacity: float = remain_capacity,
            capacity: str = 'capacity',
            max_flow: float = max_flow,
            damage_db: None|dict[tuple[int,...], float] = None,
            epsilon=1e-6,
        ):
        
        logC, = scenario_logC(
            x, beta_array=beta_array, from_condition=from_condition, G=G_comp,
            od_pairs=od_pairs, remain_capacity=remain_capacity,
            capacity='capacity', max_flow= max_flow,
            damage_db=damage_db, epsilon=epsilon,
        )

        res = b*logC

        return res

    resampler = MetropolisHastings(dimension=n_br, n_chains=_N_CHAINS)
    prior = MultivariateNormal(mean=[0.0]*n_br, cov=cov)
    if n_jobs == 1:
        use_log_pdf = lambda x,b: _log_pdf_intermediate(x,b, damage_db=damage_db)
        sampler = SequentialTemperingMCMCpar(
            log_pdf_intermediate=use_log_pdf,
            distribution_reference=prior,
            save_intermediate_samples=True,
            percentage_resampling=10,
            sampler=resampler,
            nsamples=n_smp, n_jobs=n_jobs,
        )
    else:
        with mp.Manager() as manager:
            mp_dict = manager.dict(damage_db)
            try:
                use_log_pdf = lambda x,b: _log_pdf_intermediate(x,b, damage_db=mp_dict)
                sampler = SequentialTemperingMCMCpar(
                    log_pdf_intermediate=use_log_pdf,
                    distribution_reference=prior,
                    save_intermediate_samples=True,
                    percentage_resampling=10,
                    sampler=resampler,
                    nsamples=n_smp, n_jobs=n_jobs,
                )
            except KeyboardInterrupt:
                pass
            damage_db.update(mp_dict)
    
    samples = sampler.samples
    damage_condition = get_damage_state(samples, beta_array).astype(int)
    unique_condition, counts = np.unique(damage_condition, axis=0, return_counts=True)
    evidence = sampler.evidence
    print(f'evidence estimated: {evidence}')