# %%
import numpy as np
import networkx as nx
from typing import Optional

from UQpy.distributions import Normal
from type_verifier import Numpy2DBooleanArray, Numpy2DFloatArray, NumpyFloatArray, Numpy2DIntArray


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
        flow, flow_dict = nx.maximum_flow(G1, od[0], od[1], capacity=capacity)
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
    import os
    import pickle
    import numpy as np
    import multiprocess as mp
    from datetime import datetime
    import time

    from UQpy.sampling import MetropolisHastings
    from UQpy.distributions import Normal, Uniform, MultivariateNormal
    from parallel_tempMCMC import SequentialTemperingMCMCpar

    remain_capacity = 1/60*20
    min_beta, max_beta = 0, 3
    mcmc_covar = 0.5**2
    cov_threshold = 0.2
    random_state = 42

    mode = 'run'   # or 'test' to consider only 5 bridges

    G_comp = nx.read_graphml('./assets/or_comp_graph.graphml', node_type=int)
    od_data = np.load('./assets/bnd_od.npz')
    od_pairs = od_data['bnd_od']

    # assets/damage_netdb.pkl is from run tmp/tmp_2023-07-20_05_47/damage_netdb.pkl
    with open('./assets/damage_netdb.pkl', 'rb') as f_read:
        damage_netdb = pickle.load(f_read)

    if mode == 'test':
        n_jobs = 8
        n_chains, resample_pct, n_smp = 10, 10, 100
        n_burn, n_jump = 10, 1

        n_br = 5
        max_flow = 100.0    # assumed max_flow so all tested bridges will have a non-zeros cost

        # test with the first 5 bridges
        for u, v, b in G_comp.edges.data('bridge_id'):
            if b > n_br:
                G_comp[u][v]['bridge_id'] = 0
        bridge_list = [b for _, _, b in G_comp.edges.data('bridge_id') if b>0]

    elif mode == 'run':
        n_jobs = 80
        n_chains, resample_pct, n_smp = 80, 10, 8000
        n_burn, n_jump = 1000, 10

        bridge_list = [b for _, _, b in G_comp.edges.data('bridge_id') if b>0]
        n_br = len(bridge_list)
        max_flow = net_capacity(G_comp, od_pairs=od_pairs, capacity='capacity')
    
    else:
        raise RuntimeError("Unknown mode (must be 'test' or 'run')")
    
    assert len(bridge_list) == max(bridge_list), "maximum bridge must equal to num. of bridges"

    if mode == 'test':
        damage_db = dict()
        for n in range(1, n_br+1):
            if (n,) in damage_netdb.keys():
                damage_db[(n,)] = damage_netdb[(n,)]
    elif mode == 'run':
        damage_db = dict()


    beta_array = Uniform(loc=min_beta, scale=max_beta-min_beta,).rvs(
        nsamples=n_br, random_state=1)
    beta_array = beta_array.flatten()
    pf_array = Normal().cdf(-beta_array)

    # define intermediate function
    def log_pdf_intermediate(
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

        if b == 0:
            res = np.zeros(x.shape[0])
        else:
            logC, = scenario_logC(
                x, beta_array=beta_array, from_condition=from_condition, G=G,
                od_pairs=od_pairs, remain_capacity=remain_capacity,
                capacity=capacity, max_flow= max_flow,
                damage_db=damage_db, epsilon=epsilon,
            )

            res = b*logC

        return res

    mcmc_sampler = MetropolisHastings(
        dimension=n_br, n_chains=n_chains,
        proposal=MultivariateNormal(mean=[0.0]*n_br, cov=mcmc_covar),
        burn_length=n_burn, jump=n_jump,
    )
    prior = MultivariateNormal(mean=[0.0]*n_br, cov=1.0)
    if n_jobs == 1:
        use_log_pdf = lambda x,b: log_pdf_intermediate(x,b, damage_db=damage_db)
        sampler = SequentialTemperingMCMCpar(
            log_pdf_intermediate=use_log_pdf,
            distribution_reference=prior,
            save_intermediate_samples=True,
            percentage_resampling=resample_pct,
            sampler=mcmc_sampler,
            weight_cov_threshold=cov_threshold,
            random_state=random_state,
            nsamples=n_smp, n_jobs=1,
        )
        time0 = time.time()
        sampler.run(nsamples=n_smp)
        time1 = time.time()
    else:
        with mp.Manager() as manager:
            mp_dict = manager.dict(damage_db)
            try:
                use_log_pdf = lambda x,b: log_pdf_intermediate(x,b, damage_db=mp_dict)
                prior_log_pdf = lambda x: prior.log_pdf(x)
                sampler = SequentialTemperingMCMCpar(
                    log_pdf_intermediate=use_log_pdf,
                    distribution_reference=prior,
                    save_intermediate_samples=True,
                    percentage_resampling=resample_pct,
                    sampler=mcmc_sampler,
                    weight_cov_threshold=cov_threshold,
                    random_state=random_state,
                    nsamples=n_smp, n_jobs=n_jobs,
                )
                time0 = time.time()
                sampler.parallel_run(
                    nsamples=n_smp, n_jobs=n_jobs,
                    log_pdf_intermediate=use_log_pdf,
                    prior_log_pdf=prior_log_pdf,
                )
                time1 = time.time()
            except KeyboardInterrupt:
                pass
            damage_db.update(mp_dict)
    
    samples = sampler.samples
    damage_condition = get_damage_state(samples, beta_array).astype(int)
    unique_condition, counts = np.unique(damage_condition, axis=0, return_counts=True)
    evidence = sampler.evidence
    print(f'Sampling completed in {time1-time0} seconds')
    print(f'evidence estimated: {evidence}')

    if mode == 'run':
        time_stamp = datetime.now().strftime('%Y-%m-%d_%H_%M')
        data_dir = os.path.join('./tmp', f'tmp_{time_stamp}')
        os.makedirs(data_dir, exist_ok=False)

        with open(os.path.join(data_dir, 'damage_netdb.pkl'), 'wb') as f:
            pickle.dump(damage_db, f)