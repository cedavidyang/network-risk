#%%
#    Script to calculate risk as set up in case II of the paper.
#    To reproduce the results in the paper, change the parameters in the first section.
#    Random states used in the paper are 1 to 10 for the ten runs.
#    The resutls are provided in results folder.

import numpy as np
import time

from UQpy.sampling import MetropolisHastings
from UQpy.distributions import Normal, Uniform, MultivariateNormal

from netrisk_TMCMC.sampler import SequentialTemperingMCMCpar
from netrisk_TMCMC.benchmark import scenario_logC, precise_risk, log_pdf_intermediate
from netrisk_TMCMC.net_util import get_damage_state

if __name__ == '__main__':
    # Change these parameters to test different scenarios
    n_br = 30
    useful_br = 3
    random_state = 1
    cost_type = 'swan'

    # Constants
    n_jobs = 1
    min_beta, max_beta = 0 , 3
    cost_base = 1.1    # when cost_type='swan', having a cost_base larger than 1 improves resampling process
    eps = 1e-6
    n_chains, resample_pct, n_smp = 100, 10, 5000
    n_burn, n_jump = 1000, 10
    covar = 0.5**2

    # Generate beta and cost arrays
    beta_array = Uniform(loc=min_beta, scale=max_beta-min_beta,).rvs(
        nsamples=n_br, random_state=1)
    beta_array = beta_array.flatten()
    beta_array.sort()
    pf_array = Normal().cdf(-beta_array)
    cost_array = np.power(10**beta_array, np.linspace(0.0, 1.0, n_br))
    cost_array[:-useful_br] = 1e-3

    # Evaluate precise risk
    risk0 = precise_risk(pf_array, cost_array, cost_type=cost_type, cost_base=cost_base)
    print(f"Precise risk for {cost_type} risk: {useful_br} relevant out of {n_br} assets = {risk0}\n")



    # TMCMC risk estimation

    # handle small resample fractions
    n_samples_per_chain = np.floor( ((1 - resample_pct/100)*n_smp)/n_chains ).astype(int)
    n_resamples = int(n_smp - (n_samples_per_chain * n_chains))
    if n_resamples < n_chains:
        n_chains = n_resamples

    # set up prior and intermediate distributions
    damage_db = dict()
    prior = MultivariateNormal(mean=[0.0]*n_br, cov=1.0)
    prior_log_pdf = lambda x: prior.log_pdf(x)
    use_log_pdf = lambda x, b: log_pdf_intermediate(
        x, b, beta_array=beta_array,
        cost_type=cost_type, cost_array=cost_array, cost_base=cost_base,
        damage_db=damage_db, epsilon=eps
    )

    # set up TMCMC sampler
    mcmc_sampler = MetropolisHastings(
        dimension=n_br, n_chains=n_chains,
        proposal=MultivariateNormal(mean=[0.0]*n_br, cov=covar),
        burn_length=n_burn, jump=n_jump,
    )
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

    # run sampler and time it
    time0 = time.time()
    if n_jobs == 1:
        sampler.run(nsamples=n_smp)
    else:
        sampler.parallel_run(
            nsamples=n_smp, n_jobs=n_jobs,
            log_pdf_intermediate=use_log_pdf,
            prior_log_pdf=prior_log_pdf
        )
    time1 = time.time()
    runtime_TMCMC = time1-time0

    # retrieve damage condition and unique damage condition from last-stage samples
    samples = sampler.samples
    damage_condition = get_damage_state(samples, beta_array).astype(int)
    unique_condition, counts = np.unique(damage_condition, axis=0, return_counts=True)
    evidence = sampler.evidence

    TMCMC_risk = evidence[0] - cost_base
    total_analysis = len(list(damage_db))
    all_smp = np.vstack(sampler.intermediate_samples)
    total_nsmp = all_smp.shape[0]*n_jump
    total_time = runtime_TMCMC*n_jobs

    # print results
    print("=========================")
    print("TMCMC results")
    print(f"Estimated risk = {TMCMC_risk}")
    print(f"Number of unique analysis = {total_analysis}")
    print(f"Total number of samples in all stages = {total_nsmp}")
    print(f"Total single-core run time = {total_time}\n")



    # MC risk estimation
    
    # set up comparison criteria
    max_analysis = total_analysis if total_analysis > 0 else 2**n_br
    max_smp = total_nsmp
    max_time = total_time

    # run MC
    condition_rv = MultivariateNormal(mean=[0.0]*n_br, cov=1.0)
    condition_rvs_max = condition_rv.rvs(nsamples=max_smp, random_state=random_state)

    damage_db_MC = dict()
    C_smp_list = []
    time0 = time.time()
    for i, condition_rvs in enumerate(condition_rvs_max):
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

        runtime_MC = time1-time0
        n_analysis = len(list(damage_db_MC))
        if (n_analysis >= max_analysis) or (runtime_MC>max_time):
            break

    # extract and print results
    MC_risk = np.mean(C_smp_list)
    nsmp_MC = len(C_smp_list)
    print("=========================")
    print("MCS results")
    print(f"Estimated risk = {MC_risk}")
    print(f"Number of unique analysis = {n_analysis}")
    print(f"Total number of samples = {nsmp_MC}")
    print(f"Total single-core run time = {runtime_MC}\n")
