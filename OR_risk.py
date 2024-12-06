# %%
import os
import pickle
import numpy as np
import networkx as nx
import multiprocess as mp
from datetime import datetime
import time

from typing import Optional
from type_verifier import Numpy2DBooleanArray, Numpy2DFloatArray, NumpyFloatArray, Numpy2DIntArray

from UQpy.sampling import MetropolisHastings
from UQpy.distributions import Normal, Uniform, MultivariateNormal

from parallel_tempMCMC import SequentialTemperingMCMCpar
from highway_risk_temper import net_capacity
from highway_risk_temper import scenario_logC
from highway_risk_temper import get_damage_state


if __name__ == '__main__':
    remain_capacity = 1/60*20
    min_beta, max_beta = 0, 3
    mcmc_covar = 0.5**2
    cov_threshold = 0.2
    random_state = 1
    randbeta = 1
    for random_state in random_state:    
        warm_start = True
        if warm_start:
            with open('./tmp/ttmp_2023-11-12_13_09_rand_1/damage_netdb.pkl', 'rb') as f:
                damage_db = pickle.load(f)
                print("This run will use previous damage db")
        else:
            damage_db = dict()

        mode = 'run'   # 'test' to consider only 5 bridges

        G_ = nx.read_graphml('./assets/or_hw_comp.graphml')
        # change graph attributes type
        G_comp = nx.DiGraph()
        for u, v, data in G_.edges(data=True):
            data['length'] = float(data['length'])
            data['capacity'] = float(data['capacity'])
            data['lane'] = int(data['lane'])
            data['speed'] = float(data['speed'])
            data['bridge_id'] = int(data['bridge_id'])
            G_comp.add_edge(u, v, **data)
        for n, data in G_.nodes(data=True):
            G_comp.add_node(n, **data)
        mapping = {n: int(n) for n in G_comp.nodes}
        G_comp = nx.relabel_nodes(G_comp, mapping)


        od_data = np.load('./assets/bnd_od.npz')
        od_pairs = od_data['bnd_od']

        if mode == 'test':
            n_jobs = 2
            n_chains, resample_pct, n_smp = 10, 10, 100
            n_burn, n_jump = 10, 1

            n_br = 5
            max_flow = 160    # assumed max_flow so all tested bridges will have a non-zeros cost 

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


        beta_array = Uniform(loc=min_beta, scale=max_beta-min_beta,).rvs(
            nsamples=n_br, random_state=randbeta)
        beta_array = beta_array.flatten()
        pf_array = Normal().cdf(-beta_array)
        
        print(f'setting up complete')

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
        print('random_state:', random_state)

        if mode == 'run':
            time_stamp = datetime.now().strftime('%Y-%m-%d_%H_%M')
            data_dir = os.path.join('./tmp', f'tmp_{time_stamp}_beta_{randbeta}_rand_{random_state}')
            os.makedirs(data_dir, exist_ok=False)

            with open(os.path.join(data_dir, 'damage_netdb.pkl'), 'wb') as f:
                pickle.dump(damage_db, f)
            
            # save samples, damage_condition, unique_condition
            with open(os.path.join(data_dir, 'samples.pkl'), 'wb') as f:
                pickle.dump(samples, f)
            with open(os.path.join(data_dir, 'damage_condition.pkl'), 'wb') as f:
                pickle.dump(damage_condition, f)
            with open(os.path.join(data_dir, 'unique_condition.pkl'), 'wb') as f:
                pickle.dump(unique_condition, f)
            with open(os.path.join(data_dir, 'beta_array.pkl'), 'wb') as f:
                pickle.dump(beta_array, f)
            with open(os.path.join(data_dir, 'evidence.pkl'), 'wb') as f:
                pickle.dump(evidence, f)
            
