# %%
#    This script is used to estimate the performance risk of the highway network.

import os
import pickle
import numpy as np
import networkx as nx
import osmnx as ox
import multiprocess as mp
import time
import warnings

from datetime import datetime
from typing import Optional

from UQpy.sampling import MetropolisHastings
from UQpy.distributions import Normal, Uniform, MultivariateNormal

from netrisk_TMCMC.type_verifier import NumpyFloatArray
from netrisk_TMCMC.sampler import SequentialTemperingMCMCpar
from netrisk_TMCMC.net_util import net_capacity, scenario_logC, get_damage_state


if __name__ == '__main__':
    # Parameters
    remain_capacity = 1/60*20
    min_beta, max_beta = 0, 3
    mcmc_covar = 0.5**2
    cov_threshold = 0.2

    data_dir = './assets'
    res_dir = './results'
    warm_start = False
    damage_dict_path = os.path.join(data_dir, 'damage_netdb.pkl')

    seed_TMCMC = 1
    seed_beta = 1
    
    mode = 'run'   # 'test' to consider only 5 bridges

    # One can use warm_start to continue previous runs that were interrupted
    # using the samed damage_db dictionary. To do this, set warm_start to True,
    # and make sure damage_netdb.pkl exists.
    if warm_start and os.path.exists(damage_dict_path):
        with open(damage_dict_path, 'rb') as f:
            damage_db = pickle.load(f)
            warnings.warn("Using previous damage db", RuntimeWarning)
    else:
        damage_db = dict()


    # prepare grapha and od data
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


    # Set up running parameters based on running type
    if mode == 'test':
        n_jobs = 2
        n_chains, resample_pct, n_smp = 10, 10, 100
        n_burn, n_jump = 10, 1

        n_br = 5
        max_flow = 160 

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


    # Generate beta_array and pf_array
    beta_array = Uniform(loc=min_beta, scale=max_beta-min_beta,).rvs(
        nsamples=n_br, random_state=seed_beta)
    beta_array = beta_array.flatten()
    pf_array = Normal().cdf(-beta_array)

    print("=========================")
    print("Set up complete")


    # Run TMCMC
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

    # set up and run TMCMC sampler
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
            random_state=seed_TMCMC,
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
                    random_state=seed_TMCMC,
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
    

    # retrieve results and print
    samples = sampler.samples
    damage_condition = get_damage_state(samples, beta_array).astype(int)
    unique_condition, counts = np.unique(damage_condition, axis=0, return_counts=True)
    evidence = sampler.evidence
    print(f'Sampling completed in {time1-time0} seconds')
    print(f'evidence estimated: {evidence}')


    # save results
    if mode == 'run':
        time_stamp = datetime.now().strftime('%Y-%m-%d_%H_%M')
        res_subdir = os.path.join(res_dir, f'run_{time_stamp}')
        os.makedirs(res_subdir, exist_ok=False)

        np.savez(
            os.path.join(res_subdir, 'results.npz'),
            samples=samples,    # save last-stage samples for asset importance
            beta_array=beta_array,
            damage_condition=damage_condition,
            evidence=evidence
        )

        with open(os.path.join(res_subdir, 'damage_netdb.pkl'), 'wb') as f:
            pickle.dump(damage_db, f)
