#%% Import Libraries
import numpy as np
from multiprocessing import Pool, cpu_count
import time
from UQpy.distributions import Normal, Uniform, MultivariateNormal
from highway_risk_temper import get_damage_state, net_capacity, scenario_cost
import networkx as nx
import os
import pickle

# Constants
max_smp = 3200000  # Equal to the number of TMCMC simulation samples
batch_size = 50000  # Process in smaller batches to reduce memory usage


min_beta, max_beta = 0, 3

rand_state = 1

n_br = 1938


beta_array = Uniform(loc=min_beta, scale=max_beta - min_beta).rvs(nsamples=n_br, random_state=rand_state).flatten().astype(np.float32)
pf_array = Normal().cdf(-beta_array).astype(np.float32)

condition_rv = MultivariateNormal(mean=[0.0] * n_br, cov=1.0)
condition_rvs_max = condition_rv.rvs(nsamples=max_smp, random_state=rand_state).astype(np.float32)

# damage condition of the samples
damage_condition = get_damage_state(condition_rvs_max, beta_array).astype(np.float32)

