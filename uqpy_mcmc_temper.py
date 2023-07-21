# %%
import numpy as np
from UQpy.distributions import Normal

damage_db: dict[tuple[int, ...], float] = dict()

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


def scenario_cost(damage_condition, cost_array, epsilon=1e-6, damage_db=None):

    if damage_db is None:
        damage_db = dict()

    n_smps = damage_condition.shape[0]
    cost = np.zeros(n_smps)
    for i, condition in enumerate(damage_condition):
        key = tuple(np.where(condition)[0])
        if key in damage_db.keys():
            i_cost = damage_db[key]
        else:
            i_cost = np.sum(cost_array[condition]) + epsilon
            damage_db[key] = i_cost
        cost[i] = i_cost

    return cost


def scenario_logC(condition_var, beta_array, cost_array, from_condition=False, damage_db=None):

    if from_condition:
        damage_condition = condition_var.astype(bool)
    else:
        damage_condition = get_damage_state(condition_var, beta_array=beta_array)
    
    cost = scenario_cost(damage_condition, cost_array, damage_db=damage_db)
    
    logC = np.log(cost)

    return logC


def logpC_target(x, beta_array, cost_array, from_condition=False):
    if from_condition:
        damage_condition = x.astype(bool)
    else:
        damage_condition = get_damage_state(x, beta_array=beta_array)
    logp = scenario_logp(x, beta_array, from_condition=from_condition)
    logC = np.log(scenario_cost(damage_condition, cost_array=cost_array))
    log_pC = logp + logC
    return log_pC
    

if __name__ == '__main__':
    import numpy as np
    import multiprocess as mp
    from UQpy.sampling import MetropolisHastings
    from UQpy.distributions import JointIndependent, Normal, Uniform, MultivariateNormal
    from itertools import repeat

    from parallel_tempMCMC import SequentialTemperingMCMCpar

    name = 'temper'
    n_br, min_beta, max_beta, n_smp = 5, 0, 3, 1000
    beta_array = Uniform(loc=min_beta, scale=max_beta-min_beta,).rvs(
        nsamples=n_br, random_state=1)
    beta_array = beta_array.flatten()
    pf_array = Normal().cdf(-beta_array)
    cost_array = np.arange(1.0, 1.0+n_br, 1)

    _BURN_LENGTH, _JUMP, _N_CHAINS = 10000, 100, 10
    seed0 = np.ones(n_br)*(-beta_array.min())    # only the one with min beta fails
    seed1 = np.ones(n_br)*(-beta_array.max())    # all fail
    all_seed = np.vstack(list(repeat([seed0, seed1], _N_CHAINS//2)))

    cov = 1.0
    proposal = MultivariateNormal(mean=[0.0]*n_br, cov=cov)
    damage_db = dict()

    n_jobs = 4
    # start = time.time()
    if name == 'MH':
        sampler = MetropolisHastings(
            log_pdf_target=logpC_target,
            args_target=(beta_array, cost_array),
            burn_length = _BURN_LENGTH, jump=_JUMP,
            proposal=proposal,
            seed=list(all_seed), nsamples=n_smp, random_state=1,
        )
    elif name == 'temper':
        resampler = MetropolisHastings(dimension=n_br, n_chains=_N_CHAINS)
        prior = MultivariateNormal(mean=[0.0]*n_br, cov=1.0)
        if n_jobs == 1:
            damage_db = dict()
            def _log_pdf_intermediate(x, b, damage_db=damage_db):
                logC, = scenario_logC(x, beta_array, cost_array, from_condition=False, damage_db=damage_db),
                return  b*logC
            sampler = SequentialTemperingMCMCpar(
                log_pdf_intermediate=_log_pdf_intermediate,
                distribution_reference=prior,
                save_intermediate_samples=True,
                percentage_resampling=10,
                sampler=resampler,
                nsamples=n_smp, n_jobs=n_jobs
            )
            sampler.proposal_given_flag = False
        else:
            with mp.Manager() as manager:
                mp_dict = manager.dict()
                def _log_pdf_intermediate(x, b, damage_db=mp_dict):
                    logC, = scenario_logC(x, beta_array, cost_array, from_condition=False, damage_db=damage_db),
                    return  b*logC
                sampler = SequentialTemperingMCMCpar(
                    log_pdf_intermediate=_log_pdf_intermediate,
                    distribution_reference=prior,
                    save_intermediate_samples=True,
                    percentage_resampling=10,
                    sampler=resampler,
                    nsamples=n_smp, n_jobs=n_jobs
                )
                sampler.proposal_given_flag = False
                damage_db.update(mp_dict)
    # end = time.time()
    # print(f'Sampling time: {end-start}')
    
    samples = sampler.samples
    damage_condition = get_damage_state(samples, beta_array).astype(int)
    unique_condition, counts = np.unique(damage_condition, axis=0, return_counts=True)
    evidence = sampler.evidence
    print(f'evidence estimated: {evidence}')

# %%

if __name__ == '__main__':
    import itertools

    enumerate_all = False
    ntop = 10

    if enumerate_all:
        all_condition = np.array(list(itertools.product([0,1], repeat=n_br)))
        logpC_all = logpC_target(all_condition, beta_array, cost_array, from_condition=True)

    logpC_smp = logpC_target(unique_condition, beta_array, cost_array, from_condition=True)

    sorting_indx = pf_array.argsort()[::-1]
    comb_condition = []
    total_scenario = len(logpC_smp)
    nsc = 0
    for failed in range(1, n_br):
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
    logpC_comb = logpC_target(comb_condition, beta_array, cost_array, from_condition=True)

    # compare true top scenarios and sampled top scenarios
    n_unique = unique_condition.shape[0]
    top = np.minimum(ntop, n_unique)
    if enumerate_all:
        true_top = logpC_all[np.argsort(logpC_all)[:-(top+1):-1]]
        print(f'True top logpC from {len(logpC_all)} conditions: {true_top}')
    smp_top = logpC_smp[np.argsort(logpC_smp)[:-(top+1):-1]]
    print(f'Top logpC from {n_unique} unique conditions: {smp_top}')

    # error of estimation
    if enumerate_all:
        risk_precise1 = np.sum(np.exp(logpC_all))
        print(f'precise risk from enumeration = {risk_precise1}')

    risk_precise0 = pf_array @ cost_array
    risk_appro = np.sum(np.exp(logpC_smp))
    risk_bound = risk_appro + np.exp(logpC_smp.min() + n_br*np.log(2))
    risk_appro0 = np.sum(np.exp(logpC_comb))

    print(f'precise risk = {risk_precise0}')
    print(f'estimated risk from samples = {risk_appro}')
    print(f'estimation bounded by = {risk_bound}')
    print(f'estimated risk from comb = {risk_appro0}')

# %%
