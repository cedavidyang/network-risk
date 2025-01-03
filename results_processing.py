# Result Analysis

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from UQpy.distributions import Normal
import os

# load all results from the results folder and save it in a dictionary with tmcmc seed as key
parent_folder = './results'
damage_conds= {}
evidences = {}
folder_list = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]

for folder in folder_list:
    seed = int(folder.split('_')[-1])
    with open(f'{parent_folder}/{folder}/results.npz', 'rb') as f:
        damage_conds[seed] = np.load(f)['damage_condition']
        evidences[seed] = np.load(f)['evidence']

# probabilities 
beta_array = np.load(f'{parent_folder}/{folder_list[0]}/results.npz')['beta_array'] # same accross all seeds
pf_array = Normal().cdf(-beta_array)

# aggregate damage_conds into array of shape (n_seeds, n_samples, n_links)
damage_conds_array = np.array([damage_conds[seed] for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 42]])
post_pf_array = np.mean(np.mean(damage_conds_array, axis=1), axis=0)

# Save results
df = pd.DataFrame({'Link ID': np.arange(1, len(beta_array)+1), 'Prior PF': pf_array, 'Posterior PF': post_pf_array})
df.to_excel(f'{parent_folder}/posterior_pf.xlsx', index=False)

