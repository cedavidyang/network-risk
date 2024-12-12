# %%
import numpy as np
import itertools

from highway_risk_temper import get_damage_state, scenario_logp


def scenario_cost(damage_condition, cost_array,
        damage_db=None, cost_type='add', cost_base=0.0, epsilon=1e-6):

    if damage_db is None:
        damage_db = dict()

    n_smps = damage_condition.shape[0]
    cost = np.zeros(n_smps)
    for i, condition in enumerate(damage_condition):
        key = tuple(np.where(condition)[0])
        if key in damage_db:
            i_cost = damage_db[key]
        else:
            if cost_type == 'add':
                i_cost =i_cost = np.sum(cost_array[condition]) + epsilon 
            elif cost_type == 'swan':
                if np.sum(condition)>0:
                    cost_indx = (condition.astype(bool)) & (cost_array>cost_base+epsilon)
                    i_cost = cost_array[cost_indx].prod() + \
                        cost_base + epsilon
                else:
                    i_cost = cost_base + epsilon
            else:
                raise RuntimeError("cost_type must be 'add' or 'swan'")
        cost[i] = i_cost

    return cost


def scenario_logC(condition_var, beta_array, cost_array,
                  from_condition=False,
                  cost_type='add', cost_base=0.0,
                  damage_db=None, epsilon=1e-6):

    if from_condition:
        damage_condition = condition_var.astype(bool)
    else:
        damage_condition = get_damage_state(condition_var, beta_array)
    
    assert damage_condition.ndim == 2, "damage_condition must be a 2darray"
    
    cost = scenario_cost(damage_condition, cost_array, damage_db=damage_db,
                         cost_type=cost_type, cost_base=cost_base,
                         epsilon=epsilon)
    
    logC = np.log(cost)

    return logC


def logpC_target(x, beta_array, cost_array, from_condition=False,
                 damage_db=None, cost_type='add', cost_base=0.0, epsilon=1e-6):
    if from_condition:
        damage_condition = x.astype(bool)
    else:
        damage_condition = get_damage_state(x, beta_array)

    logp = scenario_logp(x, beta_array, from_condition=from_condition)
    logC = np.log(scenario_cost(damage_condition, cost_array, damage_db=damage_db,
                                cost_type=cost_type, cost_base=cost_base, epsilon=epsilon))
    log_pC = logp + logC

    return log_pC


def precise_risk(pf_array, cost_array, cost_type='add', cost_base=1.0, epsilon=1e-6):

    if cost_type == 'add':
        total_risk = pf_array @ cost_array

    elif cost_type == 'swan':
        useful_pf = pf_array[cost_array>cost_base+epsilon]
        useful_cost = cost_array[cost_array>cost_base+epsilon]

        n_case = useful_pf.shape[0]
        total_risk = 0
        for condition in itertools.product([0,1], repeat=n_case):
            if np.sum(condition)>0:
                c_case = useful_cost[np.array(condition, dtype=bool)].prod() + epsilon + cost_base
            else:
                c_case = epsilon + cost_base
            p_case = np.prod(useful_pf[np.array(condition, dtype=bool)]) * np.prod(1-useful_pf[~np.array(condition, dtype=bool)])
            total_risk += c_case*p_case
    
        total_risk -= cost_base
    
    else:
        raise RuntimeError("cost_type must be 'add' or 'swan'")
    
    return total_risk


def log_pdf_intermediate(x, b, beta_array=None, cost_array=None, cost_type='add',
                         cost_base=0.0, prior_cov=1.0, damage_db=None, epsilon=1e-6):
    logC, = scenario_logC(x, beta_array, cost_array,
                          from_condition=False,
                          cost_type=cost_type,
                          cost_base=cost_base,
                          damage_db=damage_db,
                          epsilon=epsilon),
    return  b*logC


if __name__ == '__main__':
    pass