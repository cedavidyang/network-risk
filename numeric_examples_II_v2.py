#%%
# Import
import numpy as np
import multiprocess as mp
import itertools
import time
from UQpy.sampling import MetropolisHastings
from UQpy.distributions import Normal, Uniform, MultivariateNormal
from parallel_tempMCMC import SequentialTemperingMCMCpar
from temper_benchmark import scenario_logp, scenario_logC, precise_risk, log_pdf_intermediate
from highway_risk_temper import get_damage_state

def compare_methods(n_br,useful_br, random_state):
    # Constants

    n_jobs =1
    eps = 1e-6
    min_beta, max_beta = 0 , 3+eps
    beta_array = Uniform(loc=min_beta, scale=max_beta-min_beta,).rvs(
        nsamples=n_br, random_state=1) #
    beta_array = beta_array.flatten()
    beta_array.sort()
    pf_array = Normal().cdf(-beta_array)

    # Cost
    cost_type = 'add'
    if cost_type == 'add':
        cost_array = np.arange(1.0, 1.0+n_br, 1)
        cost_base = 0.0
    elif cost_type == 'swan':
        cost_array = np.power(10**beta_array, np.linspace(0.0, 1.0, n_br))
        cost_array[:-useful_br] = 1e-3
        cost_base = 1.1    # when cost_type='swan', having a cost_base larger than 1 improves resampling process
    else:
        raise RuntimeError("Unknown cost_type: must be 'add' or 'swan'")

    # Precise risk
    risk0 = precise_risk(pf_array, cost_array, cost_type=cost_type, cost_base=cost_base)

    # TMCMC
    n_chains, resample_pct, n_smp = 100, 10, 5000
    n_burn, n_jump = 1000, 10
    covar = 0.5**2

    n_samples_per_chain = int(np.floor(((1 - resample_pct/100) * n_smp) / n_chains))
    n_resamples = int(n_smp - (n_samples_per_chain * n_chains))
    if n_resamples < n_chains: n_chains = n_resamples

    damage_db = dict()
    prior = MultivariateNormal(mean=[0.0]*n_br, cov=1.0)

    mcmc_sampler = MetropolisHastings(
        dimension=n_br, n_chains=n_chains,
        proposal=MultivariateNormal(mean=[0.0]*n_br, cov=covar),
        burn_length=n_burn, jump=n_jump,
    )

    prior_log_pdf = lambda x: prior.log_pdf(x)

    use_log_pdf = lambda x, b: log_pdf_intermediate(
        x, b, beta_array=beta_array, cost_type=cost_type, cost_array=cost_array,
        cost_base=cost_base, damage_db=damage_db, epsilon=eps)

    sampler = SequentialTemperingMCMCpar(
        log_pdf_intermediate=use_log_pdf,
        distribution_reference=prior,
        save_intermediate_samples=True,
        percentage_resampling=resample_pct,
        sampler=mcmc_sampler,
        weight_cov_threshold=0.2,
        random_state=random_state,
        nsamples=n_smp,
    )
    time0 = time.time()
    if n_jobs == 1:
        sampler.run(nsamples=n_smp)
    else:
        sampler.parallel_run(nsamples=n_smp, n_jobs=n_jobs,
                            log_pdf_intermediate=use_log_pdf,
                            prior_log_pdf=prior_log_pdf)
    time1 = time.time()

    samples = sampler.samples

    # retrieve damage condition and unique damage condition from last-stage samples
    damage_condition = get_damage_state(samples, beta_array).astype(int)
    unique_condition, counts = np.unique(damage_condition, axis=0, return_counts=True)

    evidence = sampler.evidence

    total_analysis = len(list(damage_db))
    all_smp = np.vstack(sampler.intermediate_samples)
    total_smp = all_smp.shape[0]*n_jump
    total_time = (time1-time0)*n_jobs

    no_stages = len(sampler.intermediate_samples)
    TM_risk = evidence[0] - cost_base
    TM_no_samples = no_stages*n_smp
    TM_no_evals = total_analysis if total_analysis>0 else 2**n_br
    TM_time = total_time
    TM_no_stages = no_stages
    TM_no_samples_per_stage = n_smp

    # MC
    max_analysis = total_analysis if total_analysis>0 else 2**n_br
    max_smp = total_smp
    max_time = total_time

    condition_rv = MultivariateNormal(mean=[0.0]*n_br, cov=1.0)
    condition_rvs_max = condition_rv.rvs(nsamples=max_smp, random_state=random_state)

    damage_db_MC = dict()
    C_smp_list = []
    time0 = time.time()
    for i, condition_rvs in enumerate(condition_rvs_max):
    # for i in range(max_smp):
    #     condition_rvs = condition_rv.rvs(nsamples =1, random_state=random_state)
        condition_rvs = condition_rvs.reshape((1, -1))
        logC = scenario_logC(
            condition_rvs, beta_array=beta_array,
            cost_array=cost_array,
            from_condition=False,
            cost_type=cost_type,
            cost_base=cost_base,
            damage_db=damage_db_MC,
            epsilon=eps,
        )

        C_smp = np.exp(logC).flatten()[0]
        C_smp_list.append(C_smp)
        time1 = time.time()
        n_analysis = len(list(damage_db_MC))
        if (n_analysis >= max_analysis) or (time1-time0>max_time):
            break

    C_smp_array = np.array(C_smp_list)
    MC_risk = np.mean(C_smp_array)
    MC_no_samples = C_smp_array.shape[0]
    MC_no_evaluations = len(list(damage_db_MC))
    MC_time = time1-time0
    MC_no_samples_per_stage = MC_no_samples
    MC_no_stages = 1

    # Risk-bound
    total_scenario = total_analysis if total_analysis>0 else max_smp

    sorting_indx = cost_array.argsort()[::-1]
    comb_condition = []
    nsc = 0
    for failed in range(0, n_br+1):
        for indx in itertools.combinations(sorting_indx, failed):
            key = np.zeros(n_br, dtype=bool)
            key[(indx,)] = True
            comb_condition.append(key)
            nsc += 1
            if nsc >= total_scenario:
                break
        if nsc >= total_scenario:
            break

    comb_condition = np.array(comb_condition)
    logp_comb = scenario_logp(comb_condition, beta_array, from_condition=True)
    logC_comb = scenario_logC(comb_condition, beta_array, cost_array,
                            from_condition=True, cost_type=cost_type,
                            cost_base=cost_base, damage_db=None, epsilon=eps)

    all_fail = np.ones((1, n_br), dtype=bool)
    logC_max = scenario_logC(all_fail, beta_array, cost_array,
                            from_condition=True, cost_type=cost_type,
                            cost_base=cost_base, damage_db=None, epsilon=eps)[0]

    logC_min = np.min(cost_array.min())

    logpC_comb = logp_comb + logC_comb
    risk_bound0 = np.sum(np.exp(logpC_comb))
    p_remain = np.maximum(0, 1-np.exp(logp_comb).sum())
    risk_bound1 = risk_bound0 + p_remain*np.exp(logC_min)
    risk_bound2 = risk_bound0 + p_remain*np.exp(logC_max)

    RB_risk = risk_bound0
    RB_no_samples = -99
    RB_no_evaluations = total_scenario
    RB_time = -99
    RB_no_stages = -99
    RB_no_samples_per_stage = -99

    results = {
        'TMCMC': [TM_risk, TM_time, TM_no_evals, TM_no_samples, TM_no_samples_per_stage,   TM_no_stages],
        'MC': [MC_risk, MC_time, MC_no_evaluations, MC_no_samples, MC_no_samples_per_stage, MC_no_stages],
        'RB': [RB_risk, RB_time, RB_no_evaluations, RB_no_samples, RB_no_samples_per_stage, RB_no_stages],
        'benchmark': risk0,
        'input' : [n_br, random_state],
    }    
    return results


n_brs = [5, 10, 30, 50, 100]
random_states = [1, 2, 3, 4, 5]
useful_brs = [1] # doesn't matter for add cost

TM_vs_All = dict()
run = 1
for n_br in n_brs:
    for random_state in random_states:
        for useful_br in useful_brs:
            results = compare_methods(n_br,useful_br, random_state)
            TM_vs_All[run] = results
            TM_vs_All[run]['input'] = [n_br, random_state, useful_br]
            run += 1
            print('n_br = {}, random_state = {}, n_useful = {}'.format(n_br, random_state, useful_br))


#%%
# Function to calculate to extract results
def extract_results(All_results, method, n_br, n_use, idx=0):
    value = [result[method][idx] for result in All_results.values() if (result['input'][0] == n_br) and (result['input'][-1] == n_use)]
    return value

# Function to calculate the benchmark
def get_benchmark(All_results, n_br, n_use):
    benchmark = [result['benchmark'] for result in All_results.values() if (result['input'][0] == n_br) and (result['input'][-1] == n_use)]
    return benchmark
#%%

# Will write results to a csv file

methods = ['TMCMC', 'MC', 'RB']
with open('./TM_vs_All.csv', 'a') as f:
    f.write('Run, Random state, Dimension, Method, no. of stages, samples per stage, no. of samples, no. of evaluations, Wall-clock time, Result, Benchmark\n')
    for run in TM_vs_All.keys():
        for method in methods:
            f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                run, TM_vs_All[run]['input'][1], TM_vs_All[run]['input'][0], method,
                TM_vs_All[run][method][-1], TM_vs_All[run][method][-2], TM_vs_All[run][method][3],
                TM_vs_All[run][method][2], TM_vs_All[run][method][1], TM_vs_All[run][method][0],
                TM_vs_All[run]['benchmark']
            ))

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np

# def boxplots(dictionary, n_br, n_use):

#     # Create a figure and axis
#     fig, ax = plt.subplots()

#     # Create boxplots for TMCMC and MC columns for 30 br and 3 useful br
#     TMCMC = extract_results(dictionary, 'TMCMC', n_br, n_use)
#     MC = extract_results(dictionary, 'MC', n_br, n_use)
#     # RB = extract_results(dictionary, 'RB', n_br, n_use)
#     # RB_mean, RB_std = np.mean(RB), np.std(RB)
#     TMCMC_mean, MC_mean = np.mean(TMCMC), np.mean(MC)
#     TMCMC_std, MC_std = np.std(TMCMC), np.std(MC)
#     both = pd.DataFrame({'TMCMC': TMCMC, 'MC': MC})
#     precise = get_benchmark(dictionary, n_br, n_use)
#     sns.boxplot(data=both, ax=ax,whis=(0,100))
#     # Add a horizontal line for Precise
#     ax.axhline(y=precise[0], color='red', linestyle='--', label='Precise riks')

#     # Add points for each row and label TMCMC and MC
#     sns.stripplot(data=both, color="orange", ax=ax)
#     # Set labels and title for the main plot
#     ax.set_xlabel("Methods")
#     ax.set_ylabel("Risk")
#     ax.set_title("For {} useful and {} total bridges".format(n_use, n_br))

#     # # Label mean and std for TMCMC and MC
#     # ax.text(0, 0.5* min(np.min(TMCMC),np.min(MC)), 'Mean = {:.2f}\nStd = {:.2f}'.format(TMCMC_mean, TMCMC_std), ha='center', va='center', size=10)
#     # ax.text(1, 0.5* min(np.min(TMCMC),np.min(MC)), 'Mean = {:.2f}\nStd = {:.2f}'.format(MC_mean, MC_std), ha='center', va='center', size=10,)

#     # Show legend for the main plot
#     ax.legend()


#     # Show the plot
#     plt.savefig('./assets/{}_bridges_{}_useful.png'.format(n_br, n_use))
#     plt.show()
#     print(f'mean TMCMC = {TMCMC_mean}, mean MC = {MC_mean}')
#     print(f'std TMCMC = {TMCMC_std}, std MC = {MC_std}')
#     print(f'benchmark = {precise[0]}')

# #%% 
# # load
# import pickle
# with open('TM_vs_MC_with_RB.pickel', 'rb') as f:
#     TM_vs_All = pickle.load(f)
# #%%
# n_brs = [30, 50, 100]
# useful_brs = [3, 5, 10]
# for n_br in n_brs:
#     for n_use in useful_brs:
#         boxplots(TM_vs_All, n_br, n_use)


# #%%
# # # Save results
# # import pickle
# # with open('TM_vs_MC_with_RB.pickel', 'wb') as f:
# #     pickle.dump(TM_vs_All, f)


# %%
