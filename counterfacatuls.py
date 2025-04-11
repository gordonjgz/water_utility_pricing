import numpy as np
import pandas as pd
import statsmodels.api as sm
#from pandas.stats.api import ols
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy.stats import norm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import jax
import jax.numpy as jnp
from jax import random
from scipy.optimize import minimize
from jax.scipy.special import erf
import gc
from jax import jit, lax

#########################
##### Set Up #######
#########################


demand_2018_using = pd.read_csv('demand_2018_using.csv')

#lawn_bins = [0, 5000, 6500, 8500, 10000, np.inf]
#lawn_cat = np.digitize(demand_2018['lawn_area'], lawn_bins)
#demand_2018['lawn_cat'] = lawn_cat
#demand_2018['total_area'] = demand_2018['house_area'] + demand_2018['lawn_area']
#demand_2018['lawn_percentage'] = demand_2018['lawn_area'] / demand_2018['total_area']

#lawn_p_bins = [0, 0.75, 0.8, 0.9, 1]
#demand_2018['lawn_prop_cat']= np.digitize(demand_2018['lawn_percentage'], lawn_p_bins)
#demand_2018['lawn_areaxNDVI'] = jnp.multiply(jnp.array(demand_2018['lawn_area']), jnp.array( demand_2018['previous_NDVImyd_diff']))
#demand_2018['lawn_areaxTmax']= jnp.multiply(jnp.array(demand_2018['lawn_area']), jnp.array( demand_2018['mean_TMAX_1']))
#demand_2018['lawn_areaxPRCP']= jnp.multiply(jnp.array(demand_2018['lawn_area']), jnp.array( demand_2018['total_PRCP']))

def categorize_total_area(value):
    if value < 43560: ## 1 acerage
        return 0
    else:
        return 1

#demand_2018['above_one_acre'] = demand_2018['total_area'].apply(categorize_total_area)

def scale_array (arr):
    arr_after = jnp.where(arr > 1, jnp.log(arr) + 1, arr)  
    return arr_after

def mode(array):
    
    unique_values, counts = jnp.unique(array, return_counts=True)
    max_count_index = jnp.argmax(counts)
    return unique_values[max_count_index]
def describe(array):
    """
    Generate descriptive statistics for a NumPy array.
    Parameters:
    array (numpy.ndarray): The input array.
    
    Returns:
    dict: A dictionary containing the descriptive statistics.
    """
    description = {
        'count': array.size,
        'mean': np.mean(array),
        'std': np.std(array),
        'min': np.min(array),
        '25%': np.percentile(array, 25),
        '50%': np.median(array),
        '75%': np.percentile(array, 75),
        'max': np.max(array)
    }
    return description

scale_array_jitted = jax.jit(scale_array)

A_current_outdoor = jnp.column_stack(( 
    jnp.array(demand_2018_using['heavy_water_app_area']), 
                                      jnp.array(demand_2018_using['lawn_areaxNDVI']), 
                                      jnp.array(demand_2018_using['above_one_acre'])
                                      ))
A_current_indoor = jnp.column_stack((jnp.array(demand_2018_using['bathroom']),
                                     jnp.array(demand_2018_using['above_one_acre'])
                                       ))
A_current = jnp.column_stack((
    jnp.array(demand_2018_using['heavy_water_app']),
    jnp.array(demand_2018_using['lawn_areaxNDVI']),
    jnp.array(demand_2018_using['bathroom'])
    ))
Z_current_outdoor = jnp.column_stack((jnp.array(demand_2018_using['mean_TMAX_1']),
                                      jnp.array(demand_2018_using['IQR_TMAX_1']),
                                      jnp.array(demand_2018_using['total_PRCP']) 
                                      ,jnp.array(demand_2018_using['IQR_PRCP'])))
Z_current_indoor = jnp.array(demand_2018_using['mean_TMAX_1'])
Z_current_indoor = Z_current_indoor[:, jnp.newaxis]
G = jnp.array(demand_2018_using['previous_NDVImyd_diff'])
I = jnp.array(demand_2018_using['income'])
p0 = jnp.array(demand_2018_using['previous_essential_usage_mp'])
w_i = jnp.array(demand_2018_using['quantity'])

result = np.genfromtxt('result.csv', delimiter=',', skip_header=0)

beta = result[0]
se = result[1]

b1 = jnp.array([beta[0], beta[1], beta[2]])
b2 = jnp.array([beta[3], beta[4], beta[5], beta[6]])
c_o = beta[7]
b4 = jnp.array([beta[8], beta[9], beta[10]])
c_alpha = beta[11]
r=beta[12]
sigma_eta = beta[13]
sigma_nu = beta[14]
b8 = jnp.array([beta[15], beta[16]])
b9 = jnp.array([beta[17]])
c_i = beta[18]
alpha = jnp.exp(jnp.dot(A_current, b4)
                    + c_alpha)

demand_2018_using.loc[:, 'e_alpha'] = alpha
demand_2018_using = demand_2018_using[pd.notna(demand_2018_using['e_alpha']) & np.isfinite(demand_2018_using['e_alpha'])]

alpha_threhold = 5
demand_2018_using = demand_2018_using[demand_2018_using['e_alpha'] <= alpha_threhold]

A_current_outdoor = jnp.column_stack(( 
    jnp.array(demand_2018_using['heavy_water_app_area']), 
                                      jnp.array(demand_2018_using['lawn_areaxNDVI']), 
                                      jnp.array(demand_2018_using['above_one_acre'])
                                      ))
A_current_indoor = jnp.column_stack((jnp.array(demand_2018_using['bathroom']),
                                     jnp.array(demand_2018_using['above_one_acre'])
                                       ))
A_current = jnp.column_stack((
    jnp.array(demand_2018_using['heavy_water_app']),
    jnp.array(demand_2018_using['lawn_areaxNDVI']),
    jnp.array(demand_2018_using['bathroom'])
    ))
Z_current_outdoor = jnp.column_stack((jnp.array(demand_2018_using['mean_TMAX_1']),
                                      jnp.array(demand_2018_using['IQR_TMAX_1']),
                                      jnp.array(demand_2018_using['total_PRCP']) 
                                      ,jnp.array(demand_2018_using['IQR_PRCP'])))
Z_current_indoor = jnp.array(demand_2018_using['mean_TMAX_1'])
Z_current_indoor = Z_current_indoor[:, jnp.newaxis]
G = jnp.array(demand_2018_using['previous_NDVImyd_diff'])
I = jnp.array(demand_2018_using['income'])
p0 = jnp.array(demand_2018_using['previous_essential_usage_mp'])
w_i = jnp.array(demand_2018_using['quantity'])

def find_first_nonnegative(p):
    def cond_fun(state):
        idx, found, p = state
        return jnp.logical_and(idx < p.size, jnp.logical_not(found))

    def body_fun(state):
        idx, found, p = state
        found = jnp.logical_or(found, p[idx] >= 0)
        idx = idx + 1
        return idx, found, p

    init_state = (0, False, p)
    idx, found, _ = lax.while_loop(cond_fun, body_fun, init_state)
    
    return jnp.where(found, idx - 1, 5)

find_first_nonnegative_jit = jit(find_first_nonnegative)

#def mean_ignore_nan_inf(x):
 #   return np.nanmean(x.replace([np.inf, -np.inf], np.nan))

 
def calculate_log_w(p_l):
    fc_l = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.25+29.75])
    q_kink_l = jnp.array([2, 6, 11, 20])
    p_plus1_l = jnp.append(p_l[1:5],jnp.array([jnp.nan]) )
    d_end = jnp.cumsum( (p_l - p_plus1_l)[:4] *q_kink_l)
    d_end =  jnp.insert(d_end, 0, jnp.array([0.0]) )
    
    def calculate_dk (k):
        result = -fc_l[k] - d_end[k]
        return result
        
    calculate_dk_jitted = jax.jit(calculate_dk)
    
    def get_total_wk (beta_1, beta_2,
                  #beta_3, 
                  c_wo,
                  beta_4, 
                  #beta_5,
                  c_a,
                  rho, 
                  k, 
                  beta_8, 
                  beta_9, 
                  c_wi,
                  A_i = A_current_indoor, A_o = A_current_outdoor, Z_i = Z_current_indoor, Z_o = Z_current_outdoor, 
                  A = A_current,
                  G = jnp.array(demand_2018_using['previous_NDVImyd_diff']),
                  p = p_l, I = jnp.array(demand_2018_using['income']),
                  p0 = jnp.array(demand_2018_using['previous_essential_usage_mp']), 
                  de = jnp.array(demand_2018_using['deflator']),
                  ):
        p_k = p[k]
        d_k = calculate_dk_jitted(k)
        alpha = jnp.exp(jnp.dot(A, beta_4)
                    #+ jnp.array(beta_5*G)
                    + c_a
                    )
        rho = abs(rho)
        w_outdoor = jnp.exp(jnp.dot(A_o, beta_1) + jnp.dot(Z_o, beta_2)
                       #+ jnp.array(beta_3*G)
                       - jnp.multiply( jnp.multiply(alpha,jnp.log(p_k)), de) + 
                       jnp.multiply(rho, jnp.log(jnp.maximum(I+ jnp.multiply(d_k, de), 1e-16))) + c_wo)
        w_indoor = jnp.exp(jnp.dot(A_i, beta_8) 
                       + jnp.dot(Z_i, beta_9)
                       + c_wi
                       )
        result = jnp.log(w_outdoor + w_indoor)
        return result

    get_total_wk_jitted = jax.jit(get_total_wk)

    def get_total_wk_k (k):
        result = get_total_wk_jitted(beta_1 = b1,beta_2 = b2,
                                 c_wo = c_o,beta_4 = b4,c_a = c_alpha,
                       rho = r,
                       beta_8 = b8, 
                       beta_9 = b9,
                       c_wi = c_i,
                       k=k)
        return result

    get_total_wk_k_jitted = jax.jit(get_total_wk_k)

    log_w = jnp.column_stack((get_total_wk_k_jitted(0), get_total_wk_k_jitted(1), get_total_wk_k_jitted(2),
                    get_total_wk_k_jitted(3), get_total_wk_k_jitted(4)))
    return log_w

calculate_log_w_jitted = jax.jit(calculate_log_w)

################################################
#### Calcualte eta for each household ##########
################################################ 
p_l0 = jnp.array([2.89+0.2, 4.81+0.2, 8.34+0.2, 12.70+0.2, 14.21+0.2]) 
log_w_k_base_0 = calculate_log_w_jitted(p_l0)
log_w_k_base_0= np.nanmean(log_w_k_base_0, axis=1)
est_e = np.log(w_i) - log_w_k_base_0
q_kink_l = jnp.array([2, 6, 11, 20])

eta = np.log(w_i) - log_w_k_base_0

demand_2018_using.loc[:, 'e_eta'] = eta
demand_2018_using = demand_2018_using[pd.notna(demand_2018_using['e_eta']) & np.isfinite(demand_2018_using['e_eta'])]

demand_2018_using['mean_e_eta'] = demand_2018_using.groupby('prem_id')['e_eta'].transform('mean')

demand_2018_using = demand_2018_using.sort_values(by=['prem_id', 'bill_ym'])

demand_2018_using_eta = demand_2018_using [['prem_id', 'bill_ym', 'e_alpha','e_eta','mean_e_eta']]

demand_2018_using_eta = demand_2018_using_eta.sort_values(by=['prem_id', 'bill_ym'])

demand_2018_using_eta.to_csv('demand_2018_using_eta.csv', index=False)

plt.figure(figsize=(10, 6))
demand_2018_using['mean_e_eta'].hist(bins=100, edgecolor='black')

# Add titles and labels
plt.title('Eta Distribution')
plt.xlabel('Eta')
plt.ylabel('Frequency')

# Show the plot
plt.show()

A_current_outdoor = jnp.column_stack(( 
    jnp.array(demand_2018_using['heavy_water_app_area']), 
                                      jnp.array(demand_2018_using['lawn_areaxNDVI']), 
                                      jnp.array(demand_2018_using['above_one_acre'])
                                      ))
A_current_indoor = jnp.column_stack((jnp.array(demand_2018_using['bathroom']),
                                     jnp.array(demand_2018_using['above_one_acre'])
                                       ))
A_current = jnp.column_stack((
    jnp.array(demand_2018_using['heavy_water_app']),
    jnp.array(demand_2018_using['lawn_areaxNDVI']),
    jnp.array(demand_2018_using['bathroom'])
    ))
Z_current_outdoor = jnp.column_stack((jnp.array(demand_2018_using['mean_TMAX_1']),
                                      jnp.array(demand_2018_using['IQR_TMAX_1']),
                                      jnp.array(demand_2018_using['total_PRCP']) 
                                      ,jnp.array(demand_2018_using['IQR_PRCP'])))
Z_current_indoor = jnp.array(demand_2018_using['mean_TMAX_1'])
Z_current_indoor = Z_current_indoor[:, jnp.newaxis]
G = jnp.array(demand_2018_using['previous_NDVImyd_diff'])
I = jnp.array(demand_2018_using['income'])
p0 = jnp.array(demand_2018_using['previous_essential_usage_mp'])
w_i = jnp.array(demand_2018_using['quantity'])
alpha = jnp.exp(jnp.dot(A_current, b4)
                     + c_alpha)

def calculate_log_w(p_l):
    fc_l = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.10+29.75])
    q_kink_l = jnp.array([2, 6, 11, 20])
    p_plus1_l = jnp.append(p_l[1:5],jnp.array([jnp.nan]) )
    d_end = jnp.cumsum( (p_l - p_plus1_l)[:4] *q_kink_l)
    d_end =  jnp.insert(d_end, 0, jnp.array([0.0]) )
    
    def calculate_dk (k):
        result = -fc_l[k] - d_end[k]
        return result
        
    calculate_dk_jitted = jax.jit(calculate_dk)
    
    def get_total_wk (beta_1, beta_2,
                  #beta_3, 
                  c_wo,
                  beta_4, 
                  #beta_5,
                  c_a,
                  rho, 
                  k, 
                  beta_8, 
                  beta_9, 
                  c_wi,
                  A_i = A_current_indoor, A_o = A_current_outdoor, Z_i = Z_current_indoor, Z_o = Z_current_outdoor, 
                  A = A_current,
                  G = jnp.array(demand_2018_using['previous_NDVImyd_diff']),
                  p = p_l, I = jnp.array(demand_2018_using['income']),
                  p0 = jnp.array(demand_2018_using['previous_essential_usage_mp']), 
                  de = jnp.array(demand_2018_using['deflator']),
                  ):
        p_k = p[k]
        d_k = calculate_dk_jitted(k)
        alpha = jnp.exp(jnp.dot(A, beta_4)
                    #+ jnp.array(beta_5*G)
                    + c_a
                    )
        rho = abs(rho)
        w_outdoor = jnp.exp(jnp.dot(A_o, beta_1) + jnp.dot(Z_o, beta_2)
                       #+ jnp.array(beta_3*G)
                       - jnp.multiply( jnp.multiply(alpha,jnp.log(p_k)), de) + 
                       jnp.multiply(rho, jnp.log(jnp.maximum(I+ jnp.multiply(d_k, de), 1e-16))) + c_wo)
        w_indoor = jnp.exp(jnp.dot(A_i, beta_8) 
                       + jnp.dot(Z_i, beta_9)
                       + c_wi
                       )
        result = jnp.log(w_outdoor + w_indoor)
        return result

    get_total_wk_jitted = jax.jit(get_total_wk)

    def get_total_wk_k (k):
        result = get_total_wk_jitted(beta_1 = b1,beta_2 = b2,
                                 c_wo = c_o,beta_4 = b4,c_a = c_alpha,
                       rho = r,
                       beta_8 = b8, 
                       beta_9 = b9,
                       c_wi = c_i,
                       k=k)
        return result

    get_total_wk_k_jitted = jax.jit(get_total_wk_k)

    log_w = jnp.column_stack((get_total_wk_k_jitted(0), get_total_wk_k_jitted(1), get_total_wk_k_jitted(2),
                    get_total_wk_k_jitted(3), get_total_wk_k_jitted(4)))
    return log_w

calculate_log_w_jitted = jax.jit(calculate_log_w)

################################################
#### Calcualte CS Change from Price change #####
################################################

def find_target_index(p):
    p_l0 = jnp.array([2.89+0.2, 4.81+0.2, 8.34+0.2, 12.70+0.2, 14.21+0.2]) 
    target_index = find_first_nonnegative_jit(p_l0 - p)-1
    target_index = jnp.where(target_index<0, 0, target_index)
    return target_index

find_target_index_jitted = jax.jit(find_target_index)

@jax.jit
def nan_inf_mean_axis_0(array):
    finite_mask = jnp.isfinite(array)
    array_filtered = jnp.where(finite_mask, array, jnp.nan)
    return jnp.nanmean(array_filtered, axis=0)

@jax.jit
def nan_inf_mean_axis_1(array):
    finite_mask = jnp.isfinite(array)
    array_filtered = jnp.where(finite_mask, array, jnp.nan)
    return jnp.nanmean(array_filtered, axis=1)

def cf_cs_change_p (p):
    fc_l = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.10+29.75])
    p_l0 = jnp.array([2.89+0.2, 4.81+0.2, 8.34+0.2, 12.70+0.2, 14.21+0.2]) 
    
    target_index = find_target_index_jitted(p)

    p_l = p_l0.at[target_index].set(p)
    q_kink_l = jnp.array([2, 6, 11, 20])
    p_plus1_l = jnp.append(p_l[1:5],jnp.array([jnp.nan]) )
    d_end = jnp.cumsum( (p_l - p_plus1_l)[:4] *q_kink_l)
    d_end =  jnp.insert(d_end, 0, jnp.array([0.0]) )
    def calculate_dk (k):
        result = -fc_l[k] - d_end[k]
        return result
        
    calculate_dk_jitted = jax.jit(calculate_dk)
    log_w = calculate_log_w_jitted(p_l)
    #rng_key = random.PRNGKey(101)
    #std_dev_nu = sigma_nu
    #std_dev_eta = sigma_eta
    sim = 1000
    shape = (sim, 1)  
    nu_array = jnp.array(np.random.normal(loc=0, scale=sigma_nu, size=shape))
    eta_l = jnp.array(demand_2018_using['mean_e_eta'])
    #eta_array = jnp.array(np.random.normal(loc=0, scale=std_dev_eta, size=shape))
    #eta_nu = jnp.array(jnp.column_stack((eta_array, nu_array)))
    log_w = jnp.column_stack((log_w, eta_l))
    def apply_nu(nu_l, l_w = log_w):
        #eta = etanu[0]
        nu = nu_l[0]
        def get_log_q (log_w_k, n = nu, q_l = q_kink_l):
            log_w1 = log_w_k[0]
            log_w2 = log_w_k[1]
            log_w3 = log_w_k[2]
            log_w4 = log_w_k[3]
            log_w5 = log_w_k[4]
            e = log_w_k[5]
            log_q1 = jnp.log(q_l[0])
            log_q2 = jnp.log(q_l[1])
            log_q3 = jnp.log(q_l[2])
            log_q4 = jnp.log(q_l[3])
            conditions_w = [
                (e < log_q1 - log_w1),
                ( (e>= log_q1 - log_w1) & (e< log_q1 - log_w2)),
                ( (e>= log_q1 - log_w2) & (e< log_q2 - log_w2)),
                ((e>= log_q2 - log_w2) & (e< log_q2 - log_w3)),
                ((e>= log_q2 - log_w3) & (e< log_q3 - log_w3)),
                ((e>= log_q3 - log_w3) & (e< log_q3 - log_w4)),
                ((e>= log_q3 - log_w4) & (e< log_q4 - log_w4)),
                ((e>= log_q4 - log_w4) & (e< log_q4 - log_w5)),
                (e>= log_q4 - log_w5)
            ]  
            choices = [
                log_w1 + e + n,
                log_q1 + n,
                log_w2 + e + n,
                log_q2 + n,
                log_w3 + e + n,
                log_q3 + n,
                log_w4 + e + n,
                log_q4 + n,
                log_w5 + e + n
            ]
            result = jnp.select(conditions_w, choices)
        
            conditions_q = [
                (result < jnp.log(q_kink_l[0])) ,
                ( (result>= jnp.log(q_kink_l[0])) &  (result < jnp.log(q_kink_l[1]))),
                ( (result >= jnp.log(q_kink_l[1])) &  (result < jnp.log(q_kink_l[2]))),
                ( (result >= jnp.log(q_kink_l[2])) &  (result < jnp.log(q_kink_l[3]))),
                ( (result >= jnp.log(q_kink_l[3]))) 
            ]
            tiers = [0, 1, 2, 3, 4]
            k = jnp.select(conditions_q, tiers, default=-1)
        
            return result, k
    
        get_log_q_jitted = jax.jit(get_log_q)
        
        log_q_ti = jnp.apply_along_axis(get_log_q_jitted, axis=1, arr = l_w)
        log_q = log_q_ti[0]
        ti = log_q_ti[1]
        return log_q, ti

    apply_nu_jitted = jax.jit(apply_nu)

    sim_result_qk = jnp.apply_along_axis(apply_nu_jitted, axis=1, arr = nu_array)
    sim_result_log_q=sim_result_qk[0]
    sim_result_k=sim_result_qk[1]

    sim_result_q = jnp.exp(sim_result_log_q)
    sim_result_pk = jnp.multiply(jnp.transpose(p_l[sim_result_k]), jnp.array(demand_2018_using['deflator'])[:, jnp.newaxis])
    sim_result_Ik = jnp.transpose(calculate_dk_jitted(sim_result_k)) + jnp.array(demand_2018_using['income'])[:, jnp.newaxis]
    sim_result_q = jnp.transpose(sim_result_q)
    #sim_result_v_out = jnp.multiply(jnp.divide(-1*sim_result_q, sim_result_Ik**r), 
    #                               jnp.divide(sim_result_pk, jnp.array(1-alpha)[:, jnp.newaxis]))+ sim_result_Ik ** (1-r) / (1-r)
    sim_result_v_out = -1 * jnp.multiply(jnp.exp(jnp.dot(A_current_outdoor, b1) + jnp.dot(Z_current_outdoor, b2))[:, jnp.newaxis], 
                                         jnp.divide(jnp.power(sim_result_pk, jnp.array(1-alpha)[:, jnp.newaxis]), jnp.array(1-alpha)[:, jnp.newaxis])) + sim_result_Ik ** (1-r) / (1-r)
    v_in = jnp.multiply(-1*jnp.exp(jnp.dot(A_current_indoor, b8) 
                   + jnp.dot(Z_current_indoor, b9)
                   + c_i), p0) + jnp.array(demand_2018_using['income'])
    sim_q_result = nan_inf_mean_axis_1(sim_result_q)
    sim_v_result = nan_inf_mean_axis_1(sim_result_v_out + v_in[:, jnp.newaxis])
    #sim_v_result = jnp.mean(sim_result_v_out + v_in[:, jnp.newaxis], axis = 1)
    sim_result = jnp.column_stack((sim_q_result, sim_v_result))
    #sim_result_v_df = pd.DataFrame(jnp.column_stack((sim_v_result, sim_q_result)), columns=['sim_v', 'sim_q'])
    #sim_result_v_df['bill_ym'] = demand_2018_using['bill_ym']
    #sim_result_v_df['prem_id'] =demand_2018_using['prem_id']
    #sim_result_v = jnp.column_stack( ((jnp.array(demand_2018_using['prem_id'],dtype=jnp.float32)), (jnp.array(demand_2018_using['bill_ym'],dtype=jnp.float32 )),sim_v_result, sim_q_result))
    #columns = ['prem_id', 'bill_ym', 'sim_result_v', 'sim_result_q']
    #sim_result_v_df = pd.DataFrame(sim_result_v, columns=columns)
    #sim_result_v_df = sim_result_v_df.replace([jnp.inf, -jnp.inf], jnp.nan).dropna()
    #sim_result_v_mean = sim_result_v_df.groupby(['prem_id'])['sim_result_v'].mean()
    #print("Price is: ", p)
    jax.debug.print("Price is: {p}", p= p)
    #print("Mean Utility is: ", jnp.mean(sim_v_result))
    jax.debug.print("Mean Utility is: {x}", x= jnp.nanmean(sim_v_result))
    #gc.collect()
    return sim_result

cf_cs_change_p_jitted = jax.jit(cf_cs_change_p)

p_sequence = jnp.array((3.09, 4.09, 6.01, 9.54, 13.9, 15.41))

#cs_p_change_df = cf_cs_change_p(3.09)
#cs_p_change_df_base = cs_p_change_df.groupby(['prem_id'])['sim_result_v'].mean().reset_index()

cs_p_change = jnp.array([cf_cs_change_p_jitted(x) for x in p_sequence])

#cs_p_change = jnp.apply_along_axis(cf_cs_change_p_jitted, axis=1, arr = p_sequence)

cs_p_change_stacked = jnp.column_stack(cs_p_change)

column_names = [f'{i:.5f}' for i in p_sequence for _ in range(2)]

column_names = np.array([f'{float(s.split("_")[0]):.5f}_q' if idx % 2 == 0 else f'{float(s.split("_")[0]):.5f}_v' for idx, s in enumerate(column_names)])

cs_p_change_df = pd.DataFrame(cs_p_change_stacked, columns = column_names)

cs_p_change_df.loc[:, 'prem_id'] = demand_2018_using['prem_id']
cs_p_change_df.loc[:, 'bill_ym'] = demand_2018_using['bill_ym']

cs_p_change_df.to_csv('cf_cs_p_change.csv', index=False)

cs_p_change_df_mean_premid = cs_p_change_df.groupby('prem_id').agg({col: 'mean' for col in cs_p_change_df.columns if col not in ['prem_id', 'bill_ym']})
cs_p_change_df_mean_premid_long = pd.melt(cs_p_change_df_mean_premid.reset_index(), id_vars=['prem_id'], var_name='variable', value_name='value')
cs_p_change_df_mean_premid_long[['price', 'type']] = cs_p_change_df_mean_premid_long['variable'].str.split('_', expand=True)
cs_p_change_df_mean_premid_long = cs_p_change_df_mean_premid_long.drop('variable', axis=1)
cs_p_change_df_mean_premid_long = cs_p_change_df_mean_premid_long[['prem_id', 'price', 'type', 'value']]
cs_p_change_df_mean_premid_long.to_csv('cs_p_change_mean_premid.csv', index=False)

cs_p_change_df_mean_billym = cs_p_change_df.groupby('bill_ym').agg({col: 'mean' for col in cs_p_change_df.columns if col not in ['prem_id', 'bill_ym']})
cs_p_change_df_mean_billym_long = pd.melt(cs_p_change_df_mean_billym.reset_index(), id_vars=['bill_ym'], var_name='variable', value_name='value')
cs_p_change_df_mean_billym_long[['price', 'type']] = cs_p_change_df_mean_billym_long['variable'].str.split('_', expand=True)
cs_p_change_df_mean_billym_long = cs_p_change_df_mean_billym_long.drop('variable', axis=1)
cs_p_change_df_mean_billym_long = cs_p_change_df_mean_billym_long[['bill_ym', 'price', 'type', 'value']]
cs_p_change_df_mean_billym_long.to_csv('cs_p_change_mean_billym.csv', index=False)

################################################
#### Calcualte PS Change from Price change #####
################################################

def cf_ps_change_p (p): 
    fc_l = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.10+29.75])
    p_l0 = jnp.array([2.89+0.2, 4.81+0.2, 8.34+0.2, 12.70+0.2, 14.21+0.2]) 
   
    target_index = find_target_index_jitted(p)

    p_l = p_l0.at[target_index].set(p)
    q_kink_l = jnp.array([2, 6, 11, 20])
    p_plus1_l = jnp.append(p_l[1:5],jnp.array([jnp.nan]) )
    d_end = jnp.cumsum( (p_l - p_plus1_l)[:4] *q_kink_l)
    d_end =  jnp.insert(d_end, 0, jnp.array([0.0]) )
    #def calculate_dk (k):
     #   result = -fc_l[k] - d_end[k]
      #  return result
        
    #calculate_dk_jitted = jax.jit(calculate_dk)
    log_w = calculate_log_w_jitted(p_l)

    #rng_key = random.PRNGKey(101)
    sim = 1000
    shape = (sim, 1)  
    nu_array = jnp.array(np.random.normal(loc=0, scale=sigma_nu, size=shape))
    eta_l = jnp.array(demand_2018_using['mean_e_eta'])
    #eta_array = jnp.array(np.random.normal(loc=0, scale=std_dev_eta, size=shape))
    #eta_nu = jnp.array(jnp.column_stack((eta_array, nu_array)))
    log_w = jnp.column_stack((log_w, eta_l))

    def apply_nu(nu_l, l_w = log_w):
    #eta = etanu[0]
        nu = nu_l[0]
        def get_log_q (log_w_k, n = nu, q_l = q_kink_l):
            log_w1 = log_w_k[0]
            log_w2 = log_w_k[1]
            log_w3 = log_w_k[2]
            log_w4 = log_w_k[3]
            log_w5 = log_w_k[4]
            e = log_w_k[5]
            log_q1 = jnp.log(q_l[0])
            log_q2 = jnp.log(q_l[1])
            log_q3 = jnp.log(q_l[2])
            log_q4 = jnp.log(q_l[3])
            conditions_w = [
                (e < log_q1 - log_w1),
                ( (e>= log_q1 - log_w1) & (e< log_q1 - log_w2)),
                ( (e>= log_q1 - log_w2) & (e< log_q2 - log_w2)),
                ((e>= log_q2 - log_w2) & (e< log_q2 - log_w3)),
                ((e>= log_q2 - log_w3) & (e< log_q3 - log_w3)),
                ((e>= log_q3 - log_w3) & (e< log_q3 - log_w4)),
                ((e>= log_q3 - log_w4) & (e< log_q4 - log_w4)),
                ((e>= log_q4 - log_w4) & (e< log_q4 - log_w5)),
                (e>= log_q4 - log_w5)
            ]  
            choices = [
                log_w1 + e + n,
                log_q1 + n,
                log_w2 + e + n,
                log_q2 + n,
                log_w3 + e + n,
                log_q3 + n,
                log_w4 + e + n,
                log_q4 + n,
                log_w5 + e + n
            ]
            result = jnp.select(conditions_w, choices)
    
            conditions_q = [
                (result < jnp.log(q_kink_l[0])) ,
                ( (result>= jnp.log(q_kink_l[0])) &  (result < jnp.log(q_kink_l[1]))),
                ( (result >= jnp.log(q_kink_l[1])) &  (result < jnp.log(q_kink_l[2]))),
                ( (result >= jnp.log(q_kink_l[2])) &  (result < jnp.log(q_kink_l[3]))),
                ( (result >= jnp.log(q_kink_l[3]))) 
            ]
            tiers = [0, 1, 2, 3, 4]
            k = jnp.select(conditions_q, tiers, default=-1)
    
            return result, k

        get_log_q_jitted = jax.jit(get_log_q)
    
        log_q_ti = jnp.apply_along_axis(get_log_q_jitted, axis=1, arr = l_w)
        log_q = log_q_ti[0]
        ti = log_q_ti[1]
        return log_q, ti

    apply_nu_jitted = jax.jit(apply_nu)

    sim_result_qk = jnp.apply_along_axis(apply_nu_jitted, axis=1, arr = nu_array)
    sim_result_log_q=sim_result_qk[0]
    sim_result_k=sim_result_qk[1]

    sim_result_q = jnp.exp(sim_result_log_q)
    deflator = jnp.array(demand_2018_using['deflator'])[:, jnp.newaxis]
    
    sim_result_pk = jnp.multiply(jnp.transpose(p_l[sim_result_k]), deflator)
    sim_result_q = jnp.transpose(sim_result_q)
    sim_result_k=jnp.transpose(sim_result_k)
    q_kink_l_0 = jnp.insert(q_kink_l, 0, 0)
    sim_base_q = q_kink_l_0[sim_result_k]
    sim_extra_q = sim_result_q - sim_base_q
    sim_extra_r = jnp.multiply(sim_extra_q, sim_result_pk) ## in real dollar
    cum_r = jnp.insert(jnp.cumsum(jnp.multiply(p_l[0:4],jnp.insert(jnp.diff(q_kink_l), 0, q_kink_l[0]))),0, 0) ## in nominal dollar
    sim_base_r = jnp.multiply(cum_r[sim_result_k], deflator) ## in real dollar
    sim_variable_r = sim_base_r + sim_extra_r
    sim_fixed_r = jnp.multiply(fc_l[sim_result_k], deflator)
    sim_r = sim_variable_r + sim_fixed_r
    sim_r_result = jnp.mean(sim_r, axis = 1)
    sim_q_result = jnp.mean(sim_result_q, axis = 1)
    sim_result = jnp.column_stack((sim_q_result, sim_r_result))
    #sim_result_v_df = pd.DataFrame(jnp.column_stack((sim_v_result, sim_q_result)), columns=['sim_v', 'sim_q'])
    #sim_result_v_df['bill_ym'] = demand_2018_using['bill_ym']
    #sim_result_v_df['prem_id'] =demand_2018_using['prem_id']
    #sim_result_v = jnp.column_stack( ((jnp.array(demand_2018_using['prem_id'],dtype=jnp.float32)), (jnp.array(demand_2018_using['bill_ym'],dtype=jnp.float32 )),sim_v_result, sim_q_result))
    #columns = ['prem_id', 'bill_ym', 'sim_result_v', 'sim_result_q']
    #sim_result_v_df = pd.DataFrame(sim_result_v, columns=columns)
    #sim_result_v_df = sim_result_v_df.replace([jnp.inf, -jnp.inf], jnp.nan).dropna()
    #sim_result_v_mean = sim_result_v_df.groupby(['prem_id'])['sim_result_v'].mean()
    #print("Price is: ", p)
    jax.debug.print("Price is: {p}", p= p)
    #print("Mean Utility is: ", jnp.mean(sim_v_result))
    jax.debug.print("Mean Revenue is: {x}", x= jnp.mean(sim_r_result))
    return sim_result

cf_ps_change_p_jitted = jax.jit(cf_ps_change_p)

p_sequence = jnp.array(jnp.arange(0, 20.1, 0.1))

#cs_p_change_df = cf_cs_change_p(3.09)
#cs_p_change_df_base = cs_p_change_df.groupby(['prem_id'])['sim_result_v'].mean().reset_index()

ps_p_change = jnp.array([cf_ps_change_p_jitted(x) for x in p_sequence])

#cs_p_change = jnp.apply_along_axis(cf_cs_change_p_jitted, axis=1, arr = p_sequence)

ps_p_change_stacked = jnp.column_stack(ps_p_change)

column_names = [f'{i:.5f}' for i in p_sequence for _ in range(2)]

column_names = np.array([f'{float(s.split("_")[0]):.5f}_q' if idx % 2 == 0 else f'{float(s.split("_")[0]):.5f}_r' for idx, s in enumerate(column_names)])
ps_p_change_df = pd.DataFrame(ps_p_change_stacked, columns = column_names)

ps_p_change_df.loc[:, 'prem_id'] = demand_2018_using['prem_id']
ps_p_change_df.loc[:, 'bill_ym'] = demand_2018_using['bill_ym']

ps_p_change_df.to_csv('cf_ps_p_change.csv', index=False)

ps_p_change_df_mean_premid = ps_p_change_df.groupby('prem_id').agg({col: 'mean' for col in ps_p_change_df.columns if col not in ['prem_id', 'bill_ym']})
ps_p_change_df_mean_premid_long = pd.melt(ps_p_change_df_mean_premid.reset_index(), id_vars=['prem_id'], var_name='variable', value_name='value')
ps_p_change_df_mean_premid_long[['price', 'type']] = ps_p_change_df_mean_premid_long['variable'].str.split('_', expand=True)
ps_p_change_df_mean_premid_long = ps_p_change_df_mean_premid_long.drop('variable', axis=1)
ps_p_change_df_mean_premid_long = ps_p_change_df_mean_premid_long[['prem_id', 'price', 'type', 'value']]
ps_p_change_df_mean_premid_long.to_csv('ps_p_change_mean_premid.csv', index=False)

ps_p_change_df_mean_billym = ps_p_change_df.groupby('bill_ym').agg({col: 'mean' for col in ps_p_change_df.columns if col not in ['prem_id', 'bill_ym']})
ps_p_change_df_mean_billym_long = pd.melt(ps_p_change_df_mean_billym.reset_index(), id_vars=['bill_ym'], var_name='variable', value_name='value')
ps_p_change_df_mean_billym_long[['price', 'type']] = ps_p_change_df_mean_billym_long['variable'].str.split('_', expand=True)
ps_p_change_df_mean_billym_long = ps_p_change_df_mean_billym_long.drop('variable', axis=1)
ps_p_change_df_mean_billym_long = ps_p_change_df_mean_billym_long[['bill_ym', 'price', 'type', 'value']]
ps_p_change_df_mean_billym_long.to_csv('ps_p_change_mean_billym.csv', index=False)


################################################
#### Changing two prices at the same time #####
################################################
def cf_cs_change_pl (p_l):
    #p1 = p_row[0]
    #p2 = p_row[1]
    fc_l = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.10+29.75])
    #p_l0 = jnp.array([2.89+0.2, 4.81+0.2, 8.34+0.2, 12.70+0.2, 14.21+0.2]) 
    #target_index = find_target_index_jitted(p1)
    #p_l = p_l0.at[target_index].set(p1)
    #target_index2 = find_target_index_jitted(p2)
    #p_l = p_l.at[target_index2].set(p2)
    q_kink_l = jnp.array([2, 6, 11, 20])
    p_plus1_l = jnp.append(p_l[1:5],jnp.array([jnp.nan]) )
    d_end = jnp.cumsum( (p_l - p_plus1_l)[:4] *q_kink_l)
    d_end =  jnp.insert(d_end, 0, jnp.array([0.0]) )
    def calculate_dk (k):
        result = -fc_l[k] - d_end[k]
        return result
        
    calculate_dk_jitted = jax.jit(calculate_dk)
    log_w = calculate_log_w_jitted(p_l)
    #rng_key = random.PRNGKey(101)
    #std_dev_nu = sigma_nu
    #std_dev_eta = sigma_eta
    sim = 1000
    shape = (sim, 1)  
    nu_array = jnp.array(np.random.normal(loc=0, scale=sigma_nu, size=shape))
    eta_l = jnp.array(demand_2018_using['mean_e_eta'])
    #eta_array = jnp.array(np.random.normal(loc=0, scale=std_dev_eta, size=shape))
    #eta_nu = jnp.array(jnp.column_stack((eta_array, nu_array)))
    log_w = jnp.column_stack((log_w, eta_l))

    def apply_nu(nu_l, l_w = log_w):
        #eta = etanu[0]
        nu = nu_l[0]
        def get_log_q (log_w_k, n = nu, q_l = q_kink_l):
            log_w1 = log_w_k[0]
            log_w2 = log_w_k[1]
            log_w3 = log_w_k[2]
            log_w4 = log_w_k[3]
            log_w5 = log_w_k[4]
            e = log_w_k[5]
            log_q1 = jnp.log(q_l[0])
            log_q2 = jnp.log(q_l[1])
            log_q3 = jnp.log(q_l[2])
            log_q4 = jnp.log(q_l[3])
            conditions_w = [
                (e < log_q1 - log_w1),
                ( (e>= log_q1 - log_w1) & (e< log_q1 - log_w2)),
                ( (e>= log_q1 - log_w2) & (e< log_q2 - log_w2)),
                ((e>= log_q2 - log_w2) & (e< log_q2 - log_w3)),
                ((e>= log_q2 - log_w3) & (e< log_q3 - log_w3)),
                ((e>= log_q3 - log_w3) & (e< log_q3 - log_w4)),
                ((e>= log_q3 - log_w4) & (e< log_q4 - log_w4)),
                ((e>= log_q4 - log_w4) & (e< log_q4 - log_w5)),
                (e>= log_q4 - log_w5)
            ]  
            choices = [
                log_w1 + e + n,
                log_q1 + n,
                log_w2 + e + n,
                log_q2 + n,
                log_w3 + e + n,
                log_q3 + n,
                log_w4 + e + n,
                log_q4 + n,
                log_w5 + e + n
            ]
            result = jnp.select(conditions_w, choices)
        
            conditions_q = [
                (result < jnp.log(q_kink_l[0])) ,
                ( (result>= jnp.log(q_kink_l[0])) &  (result < jnp.log(q_kink_l[1]))),
                ( (result >= jnp.log(q_kink_l[1])) &  (result < jnp.log(q_kink_l[2]))),
                ( (result >= jnp.log(q_kink_l[2])) &  (result < jnp.log(q_kink_l[3]))),
                ( (result >= jnp.log(q_kink_l[3]))) 
            ]
            tiers = [0, 1, 2, 3, 4]
            k = jnp.select(conditions_q, tiers, default=-1)
        
            return result, k
    
        get_log_q_jitted = jax.jit(get_log_q)
        
        log_q_ti = jnp.apply_along_axis(get_log_q_jitted, axis=1, arr = l_w)
        log_q = log_q_ti[0]
        ti = log_q_ti[1]
        return log_q, ti

    apply_nu_jitted = jax.jit(apply_nu)

    sim_result_qk = jnp.apply_along_axis(apply_nu_jitted, axis=1, arr = nu_array)
    sim_result_log_q=sim_result_qk[0]
    sim_result_k=sim_result_qk[1]

    sim_result_q = jnp.exp(sim_result_log_q)
    sim_result_pk = jnp.multiply(jnp.transpose(p_l[sim_result_k]), jnp.array(demand_2018_using['deflator'])[:, jnp.newaxis])
    sim_result_Ik = jnp.transpose(calculate_dk_jitted(sim_result_k)) + jnp.array(demand_2018_using['income'])[:, jnp.newaxis]
    sim_result_q = jnp.transpose(sim_result_q)
    sim_result_v_out = -1 * jnp.multiply(jnp.exp(jnp.dot(A_current_outdoor, b1) + jnp.dot(Z_current_outdoor, b2)), 
                                         jnp.divide(jnp.power(sim_result_pk, jnp.array(1-alpha)[:, jnp.newaxis]), jnp.array(1-alpha)[:, jnp.newaxis])) + sim_result_Ik ** (1-r) / (1-r)
    #= jnp.multiply(jnp.divide(-1*sim_result_q, sim_result_Ik**r), 
     #                               jnp.divide(sim_result_pk, jnp.array(1-alpha)[:, jnp.newaxis]))+ sim_result_Ik ** (1-r) / (1-r)
    v_in = jnp.multiply(-1*jnp.exp(jnp.dot(A_current_indoor, b8) 
                   + jnp.dot(Z_current_indoor, b9)
                   + c_i), p0) + jnp.array(demand_2018_using['income'])
    sim_q_result = jnp.mean(sim_result_q, axis = 1)
    sim_v_result = jnp.mean(sim_result_v_out + v_in[:, jnp.newaxis], axis = 1)
    #sim_result = jnp.column_stack((sim_q_result, sim_v_result))
    #sim_result_v_df = pd.DataFrame(jnp.column_stack((sim_v_result, sim_q_result)), columns=['sim_v', 'sim_q'])
    #sim_result_v_df['bill_ym'] = demand_2018_using['bill_ym']
    #sim_result_v_df['prem_id'] =demand_2018_using['prem_id']
    #sim_result_v = jnp.column_stack( ((jnp.array(demand_2018_using['prem_id'],dtype=jnp.float32)), (jnp.array(demand_2018_using['bill_ym'],dtype=jnp.float32 )),sim_v_result, sim_q_result))
    #columns = ['prem_id', 'bill_ym', 'sim_result_v', 'sim_result_q']
    #sim_result_v_df = pd.DataFrame(sim_result_v, columns=columns)
    #sim_result_v_df = sim_result_v_df.replace([jnp.inf, -jnp.inf], jnp.nan).dropna()
    #sim_result_v_mean = sim_result_v_df.groupby(['prem_id'])['sim_result_v'].mean()
    #print("Price is: ", p)
    jax.debug.print("Price Vector is: {p}", p= p_l)
    #print("Mean Utility is: ", jnp.mean(sim_v_result))
    jax.debug.print("Mean Utility is: {x}", x= jnp.mean(sim_v_result))
    #gc.collect()
    return jnp.mean(sim_v_result), jnp.mean(sim_q_result)

cf_cs_change_pl_jitted = jax.jit(cf_cs_change_pl)

p_tier0_sequence = jnp.array(jnp.arange(3.1, 5.1, 0.1))
p_tier1_sequence = jnp.array(jnp.arange(5.1, 8.6, 0.1))
p_tier2_sequence = jnp.array(jnp.arange(8.6, 12.9, 0.1))
p_tier3_sequence = jnp.array(jnp.arange(12.9, 14.5, 0.1))
p_tier4_sequence = jnp.array(jnp.arange(14.5, 20.1, 0.1))

def form_two_p_seq (seq1, seq2):
    max_len = max(len(seq1), len(seq2))
    padded_seq1 = jnp.pad(seq1, (0, max_len - len(seq1)), constant_values=jnp.max(seq1))
    padded_seq2 = jnp.pad(seq2, (0, max_len - len(seq2)), constant_values=jnp.max(seq2))
    result = jnp.column_stack((padded_seq1, padded_seq2))
    return result

form_two_p_seq_jitted = jax.jit(form_two_p_seq)

#p01_sequence = form_two_p_seq_jitted(p_tier0_sequence, p_tier1_sequence)
#max_len_01 = max(len(p_tier0_sequence), len(p_tier1_sequence))
#p01_sequence = jnp.column_stack((p01_sequence, jnp.repeat(8.54, max_len_01), 
 #                               jnp.repeat(12.9, max_len_01), jnp.repeat(14.41, max_len_01)))

#p01_cs_p_change = jnp.zeros((20, demand_2018_using.shape[0], 2))

#for i in range(20):
 #   p01_cs_p_change = p01_cs_p_change.at[i].set(cf_cs_change_pl_jitted(jnp.array([3.2+i*0.1, 5.2+i*0.1, 8.54, 12.9, 14.41])))
    
    
#p34_sequence = form_two_p_seq_jitted(p_tier3_sequence, p_tier4_sequence)
#max_len_34 = max(len(p_tier3_sequence), len(p_tier4_sequence))
#p34_sequence = jnp.column_stack((jnp.repeat(3.09, max_len_34),jnp.repeat(5.01, max_len_34), jnp.repeat(8.54, max_len_34), 
 #                               p34_sequence))

#p34_cs_p_change = jnp.zeros((16, demand_2018_using.shape[0], 2))

#for i in range(16):
 #   p34_cs_p_change = p34_cs_p_change.at[i].set(cf_cs_change_pl_jitted(jnp.array([3.2, 5.2, 8.54, 13+i*0.1, 14.5+i*0.1])))
    
############################################
###### Change tier 1 and 0 #####################
########################################
p10_cs_p_change = jnp.zeros((len(p_tier1_sequence), len(p_tier0_sequence), 2))

for i in range(len(p_tier1_sequence)):
    for j in range(len(p_tier0_sequence)):
        new_values = cf_cs_change_pl_jitted(jnp.array([3.1 + j * 0.1, 5.1 + i * 0.1, 8.54, 12.9, 14.41]))
        p10_cs_p_change = p10_cs_p_change.at[i, j].set(new_values)

p10_cs_p_change_v = p10_cs_p_change[:,:,0]
np.savetxt('p10_cs_p_change_v.csv', p10_cs_p_change_v, delimiter=',')
p10_cs_p_change_q = p10_cs_p_change[:,:,1]
np.savetxt('p10_cs_p_change_q.csv', p10_cs_p_change_q, delimiter=',')
del p10_cs_p_change_v, p10_cs_p_change_q

############################################
###### Change tier 1 and 2 #####################
########################################    
    
p12_cs_p_change = jnp.zeros((len(p_tier1_sequence), len(p_tier2_sequence), 2))

for i in range(len(p_tier1_sequence)):
    for j in range(len(p_tier2_sequence)):
        new_values = cf_cs_change_pl_jitted(jnp.array([3.09, 5.1 + i * 0.1, 8.6+ j * 0.1, 12.9, 14.41]))
        p12_cs_p_change = p12_cs_p_change.at[i, j].set(new_values)

p12_cs_p_change_v = p12_cs_p_change[:,:,0]
np.savetxt('p12_cs_p_change_v.csv', p12_cs_p_change_v, delimiter=',')
p12_cs_p_change_q = p12_cs_p_change[:,:,1]
np.savetxt('p12_cs_p_change_q.csv', p12_cs_p_change_q, delimiter=',')
del p12_cs_p_change_v, p12_cs_p_change_q

############################################
###### Change tier 1 and 3 #####################
########################################   
     
p13_cs_p_change = jnp.zeros((len(p_tier1_sequence), len(p_tier3_sequence), 2))

for i in range(len(p_tier1_sequence)):
    for j in range(len(p_tier3_sequence)):
        new_values = cf_cs_change_pl_jitted(jnp.array([3.09 , 5.1 + i * 0.1, 8.54, 12.9+ j * 0.1, 14.41]))
        p13_cs_p_change = p13_cs_p_change.at[i, j].set(new_values)

p13_cs_p_change_v = p13_cs_p_change[:,:,0]
np.savetxt('p13_cs_p_change_v.csv', p13_cs_p_change_v, delimiter=',')
p13_cs_p_change_q = p13_cs_p_change[:,:,1]
np.savetxt('p13_cs_p_change_q.csv', p13_cs_p_change_q, delimiter=',')
del p13_cs_p_change_v, p13_cs_p_change_q

############################################
###### Change tier 1 and 4 #####################
########################################  
      
p14_cs_p_change = jnp.zeros((len(p_tier1_sequence), len(p_tier4_sequence), 2))

for i in range(len(p_tier1_sequence)):
    for j in range(len(p_tier4_sequence)):
        new_values = cf_cs_change_pl_jitted(jnp.array([3.09 , 5.1 + i * 0.1, 8.54, 12.9, 14.5+ j * 0.1]))
        p14_cs_p_change = p14_cs_p_change.at[i, j].set(new_values)

p14_cs_p_change_v = p14_cs_p_change[:,:,0]
np.savetxt('p14_cs_p_change_v.csv', p14_cs_p_change_v, delimiter=',')
p14_cs_p_change_q = p14_cs_p_change[:,:,1]
np.savetxt('p14_cs_p_change_q.csv', p14_cs_p_change_q, delimiter=',')
del p14_cs_p_change_v, p14_cs_p_change_q

def cf_ps_change_pl (p_l):
    #p1 = p_row[0]
    #p2 = p_row[1]
    fc_l = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.10+29.75])
    #p_l0 = jnp.array([2.89+0.2, 4.81+0.2, 8.34+0.2, 12.70+0.2, 14.21+0.2]) 
    #target_index = find_target_index_jitted(p1)
    #p_l = p_l0.at[target_index].set(p1)
    #target_index2 = find_target_index_jitted(p2)
    #p_l = p_l.at[target_index2].set(p2)
    q_kink_l = jnp.array([2, 6, 11, 20])
    p_plus1_l = jnp.append(p_l[1:5],jnp.array([jnp.nan]) )
    d_end = jnp.cumsum( (p_l - p_plus1_l)[:4] *q_kink_l)
    d_end =  jnp.insert(d_end, 0, jnp.array([0.0]) )
    #def calculate_dk (k):
        #result = -fc_l[k] - d_end[k]
        #return result
        
    #calculate_dk_jitted = jax.jit(calculate_dk)
    log_w = calculate_log_w_jitted(p_l)
   
    #rng_key = random.PRNGKey(101)
    sim = 1000
    shape = (sim, 1)  
    nu_array = jnp.array(np.random.normal(loc=0, scale=sigma_nu, size=shape))
    eta_l = jnp.array(demand_2018_using['mean_e_eta'])
    #eta_array = jnp.array(np.random.normal(loc=0, scale=std_dev_eta, size=shape))
    #eta_nu = jnp.array(jnp.column_stack((eta_array, nu_array)))
    log_w = jnp.column_stack((log_w, eta_l))

    def apply_nu(nu_l, l_w = log_w):
   #eta = etanu[0]
        nu = nu_l[0]
        def get_log_q (log_w_k, n = nu, q_l = q_kink_l):
           log_w1 = log_w_k[0]
           log_w2 = log_w_k[1]
           log_w3 = log_w_k[2]
           log_w4 = log_w_k[3]
           log_w5 = log_w_k[4]
           e = log_w_k[5]
           log_q1 = jnp.log(q_l[0])
           log_q2 = jnp.log(q_l[1])
           log_q3 = jnp.log(q_l[2])
           log_q4 = jnp.log(q_l[3])
           conditions_w = [
               (e < log_q1 - log_w1),
               ( (e>= log_q1 - log_w1) & (e< log_q1 - log_w2)),
               ( (e>= log_q1 - log_w2) & (e< log_q2 - log_w2)),
               ((e>= log_q2 - log_w2) & (e< log_q2 - log_w3)),
               ((e>= log_q2 - log_w3) & (e< log_q3 - log_w3)),
               ((e>= log_q3 - log_w3) & (e< log_q3 - log_w4)),
               ((e>= log_q3 - log_w4) & (e< log_q4 - log_w4)),
               ((e>= log_q4 - log_w4) & (e< log_q4 - log_w5)),
               (e>= log_q4 - log_w5)
           ]  
           choices = [
               log_w1 + e + n,
               log_q1 + n,
               log_w2 + e + n,
               log_q2 + n,
               log_w3 + e + n,
               log_q3 + n,
               log_w4 + e + n,
               log_q4 + n,
               log_w5 + e + n
           ]
           result = jnp.select(conditions_w, choices)
   
           conditions_q = [
               (result < jnp.log(q_kink_l[0])) ,
               ( (result>= jnp.log(q_kink_l[0])) &  (result < jnp.log(q_kink_l[1]))),
               ( (result >= jnp.log(q_kink_l[1])) &  (result < jnp.log(q_kink_l[2]))),
               ( (result >= jnp.log(q_kink_l[2])) &  (result < jnp.log(q_kink_l[3]))),
               ( (result >= jnp.log(q_kink_l[3]))) 
           ]
           tiers = [0, 1, 2, 3, 4]
           k = jnp.select(conditions_q, tiers, default=-1)
   
           return result, k

        get_log_q_jitted = jax.jit(get_log_q)
   
        log_q_ti = jnp.apply_along_axis(get_log_q_jitted, axis=1, arr = l_w)
        log_q = log_q_ti[0]
        ti = log_q_ti[1]
        return log_q, ti

    apply_nu_jitted = jax.jit(apply_nu)

    sim_result_qk = jnp.apply_along_axis(apply_nu_jitted, axis=1, arr = nu_array)
    sim_result_log_q=sim_result_qk[0]
    sim_result_k=sim_result_qk[1]

    sim_result_q = jnp.exp(sim_result_log_q)
    deflator = jnp.array(demand_2018_using['deflator'])[:, jnp.newaxis]
   
    sim_result_pk = jnp.multiply(jnp.transpose(p_l[sim_result_k]), deflator)
    sim_result_q = jnp.transpose(sim_result_q)
    sim_result_k=jnp.transpose(sim_result_k)
    q_kink_l_0 = jnp.insert(q_kink_l, 0, 0)
    sim_base_q = q_kink_l_0[sim_result_k]
    sim_extra_q = sim_result_q - sim_base_q
    sim_extra_r = jnp.multiply(sim_extra_q, sim_result_pk) ## in real dollar
    cum_r = jnp.insert(jnp.cumsum(jnp.multiply(p_l[0:4],jnp.insert(jnp.diff(q_kink_l), 0, q_kink_l[0]))),0, 0) ## in nominal dollar
    sim_base_r = jnp.multiply(cum_r[sim_result_k], deflator) ## in real dollar
    sim_variable_r = sim_base_r + sim_extra_r
    sim_fixed_r = jnp.multiply(fc_l[sim_result_k], deflator)
    sim_r = sim_variable_r + sim_fixed_r
    sim_r_result = jnp.mean(sim_r, axis = 1)
    sim_q_result = jnp.mean(sim_result_q, axis = 1)
    #sim_result = jnp.column_stack((sim_q_result, sim_r_result))
    #sim_result_v_df = pd.DataFrame(jnp.column_stack((sim_v_result, sim_q_result)), columns=['sim_v', 'sim_q'])
    #sim_result_v_df['bill_ym'] = demand_2018_using['bill_ym']
    #sim_result_v_df['prem_id'] =demand_2018_using['prem_id']
    #sim_result_v = jnp.column_stack( ((jnp.array(demand_2018_using['prem_id'],dtype=jnp.float32)), (jnp.array(demand_2018_using['bill_ym'],dtype=jnp.float32 )),sim_v_result, sim_q_result))
    #columns = ['prem_id', 'bill_ym', 'sim_result_v', 'sim_result_q']
    #sim_result_v_df = pd.DataFrame(sim_result_v, columns=columns)
    #sim_result_v_df = sim_result_v_df.replace([jnp.inf, -jnp.inf], jnp.nan).dropna()
    #sim_result_v_mean = sim_result_v_df.groupby(['prem_id'])['sim_result_v'].mean()
    #print("Price is: ", p)
    jax.debug.print("Price is: {p}", p= p_l)
    #print("Mean Utility is: ", jnp.mean(sim_v_result))
    jax.debug.print("Mean Revenue is: {x}", x= jnp.mean(sim_r_result))
    return jnp.mean(sim_r_result), jnp.mean(sim_q_result)

cf_ps_change_pl_jitted = jax.jit(cf_ps_change_pl)

############################################
###### Change tier 1 and 0 #####################
########################################
 
p10_ps_p_change = jnp.zeros((len(p_tier1_sequence), len(p_tier0_sequence), 2))

for i in range(len(p_tier1_sequence)):
    for j in range(len(p_tier0_sequence)):
        new_values = cf_ps_change_pl_jitted(jnp.array([3.1 + j * 0.1, 5.1 + i * 0.1, 8.54, 12.9, 14.41]))
        p10_ps_p_change = p10_ps_p_change.at[i, j].set(new_values)

p10_ps_p_change_v = p10_ps_p_change[:,:,0]
np.savetxt('p10_ps_p_change_v.csv', p10_ps_p_change_v, delimiter=',')
p10_ps_p_change_q = p10_ps_p_change[:,:,1]
np.savetxt('p10_ps_p_change_q.csv', p10_ps_p_change_q, delimiter=',')
del p10_ps_p_change_v, p10_ps_p_change_q

############################################
###### Change tier 1 and 2 #####################
########################################
        
p12_ps_p_change = jnp.zeros((len(p_tier1_sequence), len(p_tier2_sequence), 2))

for i in range(len(p_tier1_sequence)):
    for j in range(len(p_tier2_sequence)):
        new_values = cf_ps_change_pl_jitted(jnp.array([3.09, 5.1 + i * 0.1, 8.6+ j * 0.1, 12.9, 14.41]))
        p12_ps_p_change = p12_ps_p_change.at[i, j].set(new_values)

p12_ps_p_change_v = p12_ps_p_change[:,:,0]
np.savetxt('p12_ps_p_change_v.csv', p12_ps_p_change_v, delimiter=',')
p12_ps_p_change_q = p12_ps_p_change[:,:,1]
np.savetxt('p12_ps_p_change_q.csv', p12_ps_p_change_q, delimiter=',')
del p12_ps_p_change_v, p12_ps_p_change_q

############################################
###### Change tier 1 and 3 #####################
########################################
        
p13_ps_p_change = jnp.zeros((len(p_tier1_sequence), len(p_tier3_sequence), 2))

for i in range(len(p_tier1_sequence)):
    for j in range(len(p_tier3_sequence)):
        new_values = cf_ps_change_pl_jitted(jnp.array([3.09 , 5.1 + i * 0.1, 8.54, 12.9+ j * 0.1, 14.41]))
        p13_ps_p_change = p13_ps_p_change.at[i, j].set(new_values)

p13_ps_p_change_v = p13_ps_p_change[:,:,0]
np.savetxt('p13_ps_p_change_v.csv', p13_ps_p_change_v, delimiter=',')
p13_ps_p_change_q = p13_ps_p_change[:,:,1]
np.savetxt('p13_ps_p_change_q.csv', p13_ps_p_change_q, delimiter=',')
del p13_ps_p_change_v, p13_ps_p_change_q

############################################
###### Change tier 1 and 4 #####################
########################################
        
p14_ps_p_change = jnp.zeros((len(p_tier1_sequence), len(p_tier4_sequence), 2))

for i in range(len(p_tier1_sequence)):
    for j in range(len(p_tier4_sequence)):
        new_values = cf_ps_change_pl_jitted(jnp.array([3.09 , 5.1 + i * 0.1, 8.54, 12.9, 14.5+ j * 0.1]))
        p14_ps_p_change = p14_ps_p_change.at[i, j].set(new_values)

p14_ps_p_change_v = p14_ps_p_change[:,:,0]
np.savetxt('p14_ps_p_change_v.csv', p14_ps_p_change_v, delimiter=',')
p14_ps_p_change_q = p14_ps_p_change[:,:,1]
np.savetxt('p14_ps_p_change_q.csv', p14_ps_p_change_q, delimiter=',')
del p14_ps_p_change_v, p14_ps_p_change_q


##########################################
###### Calculate CS from q_kink change ####
################################################

def find_first_nonnegative_q(q):
    def cond_fun(state):
        idx, found, q = state
        return jnp.logical_and(idx < q.size, jnp.logical_not(found))

    def body_fun(state):
        idx, found, q = state
        found = jnp.logical_or(found, q[idx] >= 0)
        idx = idx + 1
        return idx, found, q

    init_state = (0, False, q)
    idx, found, _ = lax.while_loop(cond_fun, body_fun, init_state)
    
    return jnp.where(found, idx - 1, 4)

find_first_nonnegative_q_jit = jit(find_first_nonnegative_q)

def calculate_log_w_q(q_l):
    fc_l = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.10+29.75])
    p_l = jnp.array([2.89+0.2, 4.81+0.2, 8.34+0.2, 12.70+0.2, 14.21+0.2]) 
    q_kink_l = q_l
    p_plus1_l = jnp.append(p_l[1:5],jnp.array([jnp.nan]) )
    d_end = jnp.cumsum( (p_l - p_plus1_l)[:4] *q_kink_l)
    d_end =  jnp.insert(d_end, 0, jnp.array([0.0]) )
    
    def calculate_dk (k):
        result = -fc_l[k] - d_end[k]
        return result
        
    calculate_dk_jitted = jax.jit(calculate_dk)
    
    def get_total_wk (beta_1, beta_2,
                  #beta_3, 
                  c_wo,
                  beta_4, 
                  #beta_5,
                  c_a,
                  rho, 
                  k, 
                  beta_8, 
                  beta_9, 
                  c_wi,
                  A_i = A_current_indoor, A_o = A_current_outdoor, Z_i = Z_current_indoor, Z_o = Z_current_outdoor, 
                  A = A_current,
                  G = jnp.array(demand_2018_using['previous_NDVImyd_diff']),
                  p = p_l, I = jnp.array(demand_2018_using['income']),
                  p0 = jnp.array(demand_2018_using['previous_essential_usage_mp']), 
                  de = jnp.array(demand_2018_using['deflator']),
                  ):
        p_k = p[k]
        d_k = calculate_dk_jitted(k)
        alpha = jnp.exp(jnp.dot(A, beta_4)
                    #+ jnp.array(beta_5*G)
                    + c_a
                    )
        rho = abs(rho)
        w_outdoor = jnp.exp(jnp.dot(A_o, beta_1) + jnp.dot(Z_o, beta_2)
                       #+ jnp.array(beta_3*G)
                       - jnp.multiply( jnp.multiply(alpha,jnp.log(p_k)), de) + 
                       jnp.multiply(rho, jnp.log(jnp.maximum(I+ jnp.multiply(d_k, de), 1e-16))) + c_wo)
        w_indoor = jnp.exp(jnp.dot(A_i, beta_8) 
                       + jnp.dot(Z_i, beta_9)
                       + c_wi
                       )
        result = jnp.log(w_outdoor + w_indoor)
        return result

    get_total_wk_jitted = jax.jit(get_total_wk)

    def get_total_wk_k (k):
        result = get_total_wk_jitted(beta_1 = b1,beta_2 = b2,
                                 c_wo = c_o,beta_4 = b4,c_a = c_alpha,
                       rho = r,
                       beta_8 = b8, 
                       beta_9 = b9,
                       c_wi = c_i,
                       k=k)
        return result

    get_total_wk_k_jitted = jax.jit(get_total_wk_k)

    log_w = jnp.column_stack((get_total_wk_k_jitted(0), get_total_wk_k_jitted(1), get_total_wk_k_jitted(2),
                    get_total_wk_k_jitted(3), get_total_wk_k_jitted(4)))
    return log_w

calculate_log_w_q_jitted = jax.jit(calculate_log_w_q)

def cf_cs_change_ql (q_l):
    fc_l = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.10+29.75])
    p_l = jnp.array([2.89+0.2, 4.81+0.2, 8.34+0.2, 12.70+0.2, 14.21+0.2]) 
    
    q_kink_l = q_l
    p_plus1_l = jnp.append(p_l[1:5],jnp.array([jnp.nan]) )
    d_end = jnp.cumsum( (p_l - p_plus1_l)[:4] *q_kink_l)
    d_end =  jnp.insert(d_end, 0, jnp.array([0.0]) )
    def calculate_dk (k):
        result = -fc_l[k] - d_end[k]
        return result
        
    calculate_dk_jitted = jax.jit(calculate_dk)
    log_w = calculate_log_w_q_jitted(q_l)
    #rng_key = random.PRNGKey(101)
    #std_dev_nu = sigma_nu
    #std_dev_eta = sigma_eta
    sim = 1000
    shape = (sim, 1)  
    nu_array = jnp.array(np.random.normal(loc=0, scale=sigma_nu, size=shape))
    eta_l = jnp.array(demand_2018_using['mean_e_eta'])
    #eta_array = jnp.array(np.random.normal(loc=0, scale=std_dev_eta, size=shape))
    #eta_nu = jnp.array(jnp.column_stack((eta_array, nu_array)))
    log_w = jnp.column_stack((log_w, eta_l))

    def apply_nu(nu_l, l_w = log_w):
        #eta = etanu[0]
        nu = nu_l[0]
        def get_log_q (log_w_k, n = nu, q_l = q_kink_l):
            log_w1 = log_w_k[0]
            log_w2 = log_w_k[1]
            log_w3 = log_w_k[2]
            log_w4 = log_w_k[3]
            log_w5 = log_w_k[4]
            e = log_w_k[5]
            log_q1 = jnp.log(q_l[0])
            log_q2 = jnp.log(q_l[1])
            log_q3 = jnp.log(q_l[2])
            log_q4 = jnp.log(q_l[3])
            conditions_w = [
                (e < log_q1 - log_w1),
                ( (e>= log_q1 - log_w1) & (e< log_q1 - log_w2)),
                ( (e>= log_q1 - log_w2) & (e< log_q2 - log_w2)),
                ((e>= log_q2 - log_w2) & (e< log_q2 - log_w3)),
                ((e>= log_q2 - log_w3) & (e< log_q3 - log_w3)),
                ((e>= log_q3 - log_w3) & (e< log_q3 - log_w4)),
                ((e>= log_q3 - log_w4) & (e< log_q4 - log_w4)),
                ((e>= log_q4 - log_w4) & (e< log_q4 - log_w5)),
                (e>= log_q4 - log_w5)
            ]  
            choices = [
                log_w1 + e + n,
                log_q1 + n,
                log_w2 + e + n,
                log_q2 + n,
                log_w3 + e + n,
                log_q3 + n,
                log_w4 + e + n,
                log_q4 + n,
                log_w5 + e + n
            ]
            result = jnp.select(conditions_w, choices)
        
            conditions_q = [
                (result < jnp.log(q_kink_l[0])) ,
                ( (result>= jnp.log(q_kink_l[0])) &  (result < jnp.log(q_kink_l[1]))),
                ( (result >= jnp.log(q_kink_l[1])) &  (result < jnp.log(q_kink_l[2]))),
                ( (result >= jnp.log(q_kink_l[2])) &  (result < jnp.log(q_kink_l[3]))),
                ( (result >= jnp.log(q_kink_l[3]))) 
            ]
            tiers = [0, 1, 2, 3, 4]
            k = jnp.select(conditions_q, tiers, default=-1)
        
            return result, k
    
        get_log_q_jitted = jax.jit(get_log_q)
        
        log_q_ti = jnp.apply_along_axis(get_log_q_jitted, axis=1, arr = l_w)
        log_q = log_q_ti[0]
        ti = log_q_ti[1]
        return log_q, ti

    apply_nu_jitted = jax.jit(apply_nu)

    sim_result_qk = jnp.apply_along_axis(apply_nu_jitted, axis=1, arr = nu_array)
    sim_result_log_q=sim_result_qk[0]
    sim_result_k=sim_result_qk[1]

    sim_result_q = jnp.exp(sim_result_log_q)
    sim_result_pk = jnp.multiply(jnp.transpose(p_l[sim_result_k]), jnp.array(demand_2018_using['deflator'])[:, jnp.newaxis])
    sim_result_Ik = jnp.transpose(calculate_dk_jitted(sim_result_k)) + jnp.array(demand_2018_using['income'])[:, jnp.newaxis]
    sim_result_q = jnp.transpose(sim_result_q)
    sim_result_v_out = -1 * jnp.multiply(jnp.exp(jnp.dot(A_current_outdoor, b1) + jnp.dot(Z_current_outdoor, b2))[:, jnp.newaxis], 
                                         jnp.divide(jnp.power(sim_result_pk, jnp.array(1-alpha)[:, jnp.newaxis]), jnp.array(1-alpha)[:, jnp.newaxis])) + sim_result_Ik ** (1-r) / (1-r)
    v_in = jnp.multiply(-1*jnp.exp(jnp.dot(A_current_indoor, b8) 
                   + jnp.dot(Z_current_indoor, b9)
                   + c_i), p0) + jnp.array(demand_2018_using['income'])
    sim_q_result = nan_inf_mean_axis_1(sim_result_q)
    sim_v_result = nan_inf_mean_axis_1(sim_result_v_out + v_in[:, jnp.newaxis])
    #sim_v_result = jnp.mean(sim_result_v_out + v_in[:, jnp.newaxis], axis = 1)
    sim_result = jnp.column_stack((sim_q_result, sim_v_result))
    #sim_result_v_df = pd.DataFrame(jnp.column_stack((sim_v_result, sim_q_result)), columns=['sim_v', 'sim_q'])
    #sim_result_v_df['bill_ym'] = demand_2018_using['bill_ym']
    #sim_result_v_df['prem_id'] =demand_2018_using['prem_id']
    #sim_result_v = jnp.column_stack( ((jnp.array(demand_2018_using['prem_id'],dtype=jnp.float32)), (jnp.array(demand_2018_using['bill_ym'],dtype=jnp.float32 )),sim_v_result, sim_q_result))
    #columns = ['prem_id', 'bill_ym', 'sim_result_v', 'sim_result_q']
    #sim_result_v_df = pd.DataFrame(sim_result_v, columns=columns)
    #sim_result_v_df = sim_result_v_df.replace([jnp.inf, -jnp.inf], jnp.nan).dropna()
    #sim_result_v_mean = sim_result_v_df.groupby(['prem_id'])['sim_result_v'].mean()
    #print("Price is: ", p)
    jax.debug.print("Q_kink is: {q_l}", q_l= q_l)
    #print("Mean Utility is: ", jnp.mean(sim_v_result))
    jax.debug.print("Mean Utility is: {x}", x= jnp.nanmean(sim_v_result))
    #gc.collect()
    return sim_result

cf_cs_change_ql_jitted = jax.jit(cf_cs_change_ql)

def find_target_index_q(q):
    q_kink_l = jnp.array([2.0, 6.0, 11.0, 20.0])
    target_index = find_first_nonnegative_q_jit(q_kink_l - q)
    target_index = jnp.where(target_index<0, 0, target_index)
    target_index = jnp.where(target_index>3, 3, target_index)
    return target_index

find_target_index_q_jitted = jax.jit(find_target_index_q)

def cf_cs_change_q (q):
    q_kink_l = jnp.array([2.0, 6.0, 11.0, 20.0])
    target_index = find_target_index_q_jitted(q)
    q_l = q_kink_l.at[target_index].set(q)
    return cf_cs_change_ql_jitted(q_l)

cf_cs_change_q_jitted = jax.jit(cf_cs_change_q)

q_sequence = jnp.array((1, 5, 10, 19, 20))

#cs_p_change_df = cf_cs_change_p(3.09)
#cs_p_change_df_base = cs_p_change_df.groupby(['prem_id'])['sim_result_v'].mean().reset_index()

cs_q_change = jnp.array([cf_cs_change_q_jitted(x) for x in q_sequence])

#cs_p_change = jnp.apply_along_axis(cf_cs_change_p_jitted, axis=1, arr = p_sequence)

cs_q_change_stacked = jnp.column_stack(cs_q_change)

column_names = [f'{i:.5f}' for i in q_sequence for _ in range(2)]

column_names = np.array([f'{float(s.split("_")[0]):.5f}_q' if idx % 2 == 0 else f'{float(s.split("_")[0]):.5f}_v' for idx, s in enumerate(column_names)])
cs_q_change_df = pd.DataFrame(cs_q_change_stacked, columns = column_names)

cs_q_change_df.loc[:, 'prem_id'] = demand_2018_using['prem_id']
cs_q_change_df.loc[:, 'bill_ym'] = demand_2018_using['bill_ym']

cs_q_change_df.to_csv('cf_cs_q_change.csv', index=False)

cs_q_change_df_mean_premid = cs_q_change_df.groupby('prem_id').agg({col: 'mean' for col in cs_q_change_df.columns if col not in ['prem_id', 'bill_ym']})
cs_q_change_df_mean_premid_long = pd.melt(cs_q_change_df_mean_premid.reset_index(), id_vars=['prem_id'], var_name='variable', value_name='value')
cs_q_change_df_mean_premid_long[['quantity', 'type']] = cs_q_change_df_mean_premid_long['variable'].str.split('_', expand=True)
cs_q_change_df_mean_premid_long = cs_q_change_df_mean_premid_long.drop('variable', axis=1)
cs_q_change_df_mean_premid_long = cs_q_change_df_mean_premid_long[['prem_id', 'quantity', 'type', 'value']]
cs_q_change_df_mean_premid_long.to_csv('cs_q_change_mean_premid.csv', index=False)

cs_q_change_df_mean_billym = cs_q_change_df.groupby('bill_ym').agg({col: 'mean' for col in cs_q_change_df.columns if col not in ['prem_id', 'bill_ym']})
cs_q_change_df_mean_billym_long = pd.melt(cs_q_change_df_mean_billym.reset_index(), id_vars=['bill_ym'], var_name='variable', value_name='value')
cs_q_change_df_mean_billym_long[['quantity', 'type']] = cs_q_change_df_mean_billym_long['variable'].str.split('_', expand=True)
cs_q_change_df_mean_billym_long = cs_q_change_df_mean_billym_long.drop('variable', axis=1)
cs_q_change_df_mean_billym_long = cs_q_change_df_mean_billym_long[['bill_ym', 'quantity', 'type', 'value']]
cs_q_change_df_mean_billym_long.to_csv('cs_q_change_mean_billym.csv', index=False)


##########################################
###### Calculate PS from q_kink change ####
################################################

def cf_ps_change_ql (q_l): 
    fc_l = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.10+29.75])
    p_l0 = jnp.array([2.89+0.2, 4.81+0.2, 8.34+0.2, 12.70+0.2, 14.21+0.2]) 

    p_l = p_l0
    q_kink_l = q_l
    p_plus1_l = jnp.append(p_l[1:5],jnp.array([jnp.nan]) )
    d_end = jnp.cumsum( (p_l - p_plus1_l)[:4] *q_kink_l)
    d_end =  jnp.insert(d_end, 0, jnp.array([0.0]) )
    #def calculate_dk (k):
     #   result = -fc_l[k] - d_end[k]
      #  return result
        
    #calculate_dk_jitted = jax.jit(calculate_dk)
    log_w = calculate_log_w_q_jitted(q_l)

    #rng_key = random.PRNGKey(101)
    sim = 1000
    shape = (sim, 1)  
    nu_array = jnp.array(np.random.normal(loc=0, scale=sigma_nu, size=shape))
    eta_l = jnp.array(demand_2018_using['mean_e_eta'])
    #eta_array = jnp.array(np.random.normal(loc=0, scale=std_dev_eta, size=shape))
    #eta_nu = jnp.array(jnp.column_stack((eta_array, nu_array)))
    log_w = jnp.column_stack((log_w, eta_l))

    def apply_nu(nu_l, l_w = log_w):
    #eta = etanu[0]
        nu = nu_l[0]
        def get_log_q (log_w_k, n = nu, q_l = q_kink_l):
            log_w1 = log_w_k[0]
            log_w2 = log_w_k[1]
            log_w3 = log_w_k[2]
            log_w4 = log_w_k[3]
            log_w5 = log_w_k[4]
            e = log_w_k[5]
            log_q1 = jnp.log(q_l[0])
            log_q2 = jnp.log(q_l[1])
            log_q3 = jnp.log(q_l[2])
            log_q4 = jnp.log(q_l[3])
            conditions_w = [
                (e < log_q1 - log_w1),
                ( (e>= log_q1 - log_w1) & (e< log_q1 - log_w2)),
                ( (e>= log_q1 - log_w2) & (e< log_q2 - log_w2)),
                ((e>= log_q2 - log_w2) & (e< log_q2 - log_w3)),
                ((e>= log_q2 - log_w3) & (e< log_q3 - log_w3)),
                ((e>= log_q3 - log_w3) & (e< log_q3 - log_w4)),
                ((e>= log_q3 - log_w4) & (e< log_q4 - log_w4)),
                ((e>= log_q4 - log_w4) & (e< log_q4 - log_w5)),
                (e>= log_q4 - log_w5)
            ]  
            choices = [
                log_w1 + e + n,
                log_q1 + n,
                log_w2 + e + n,
                log_q2 + n,
                log_w3 + e + n,
                log_q3 + n,
                log_w4 + e + n,
                log_q4 + n,
                log_w5 + e + n
            ]
            result = jnp.select(conditions_w, choices)
    
            conditions_q = [
                (result < jnp.log(q_kink_l[0])) ,
                ( (result>= jnp.log(q_kink_l[0])) &  (result < jnp.log(q_kink_l[1]))),
                ( (result >= jnp.log(q_kink_l[1])) &  (result < jnp.log(q_kink_l[2]))),
                ( (result >= jnp.log(q_kink_l[2])) &  (result < jnp.log(q_kink_l[3]))),
                ( (result >= jnp.log(q_kink_l[3]))) 
            ]
            tiers = [0, 1, 2, 3, 4]
            k = jnp.select(conditions_q, tiers, default=-1)
    
            return result, k

        get_log_q_jitted = jax.jit(get_log_q)
    
        log_q_ti = jnp.apply_along_axis(get_log_q_jitted, axis=1, arr = l_w)
        log_q = log_q_ti[0]
        ti = log_q_ti[1]
        return log_q, ti

    apply_nu_jitted = jax.jit(apply_nu)

    sim_result_qk = jnp.apply_along_axis(apply_nu_jitted, axis=1, arr = nu_array)
    sim_result_log_q=sim_result_qk[0]
    sim_result_k=sim_result_qk[1]

    sim_result_q = jnp.exp(sim_result_log_q)
    deflator = jnp.array(demand_2018_using['deflator'])[:, jnp.newaxis]
    
    sim_result_pk = jnp.multiply(jnp.transpose(p_l[sim_result_k]), deflator)
    sim_result_q = jnp.transpose(sim_result_q)
    sim_result_k=jnp.transpose(sim_result_k)
    q_kink_l_0 = jnp.insert(q_kink_l, 0, 0)
    sim_base_q = q_kink_l_0[sim_result_k]
    sim_extra_q = sim_result_q - sim_base_q
    sim_extra_r = jnp.multiply(sim_extra_q, sim_result_pk) ## in real dollar
    cum_r = jnp.insert(jnp.cumsum(jnp.multiply(p_l[0:4],jnp.insert(jnp.diff(q_kink_l), 0, q_kink_l[0]))),0, 0) ## in nominal dollar
    sim_base_r = jnp.multiply(cum_r[sim_result_k], deflator) ## in real dollar
    sim_variable_r = sim_base_r + sim_extra_r
    sim_fixed_r = jnp.multiply(fc_l[sim_result_k], deflator)
    sim_r = sim_variable_r + sim_fixed_r
    sim_r_result = jnp.mean(sim_r, axis = 1)
    sim_q_result = jnp.mean(sim_result_q, axis = 1)
    sim_result = jnp.column_stack((sim_q_result, sim_r_result))
    #sim_result_v_df = pd.DataFrame(jnp.column_stack((sim_v_result, sim_q_result)), columns=['sim_v', 'sim_q'])
    #sim_result_v_df['bill_ym'] = demand_2018_using['bill_ym']
    #sim_result_v_df['prem_id'] =demand_2018_using['prem_id']
    #sim_result_v = jnp.column_stack( ((jnp.array(demand_2018_using['prem_id'],dtype=jnp.float32)), (jnp.array(demand_2018_using['bill_ym'],dtype=jnp.float32 )),sim_v_result, sim_q_result))
    #columns = ['prem_id', 'bill_ym', 'sim_result_v', 'sim_result_q']
    #sim_result_v_df = pd.DataFrame(sim_result_v, columns=columns)
    #sim_result_v_df = sim_result_v_df.replace([jnp.inf, -jnp.inf], jnp.nan).dropna()
    #sim_result_v_mean = sim_result_v_df.groupby(['prem_id'])['sim_result_v'].mean()
    #print("Price is: ", p)
    jax.debug.print("Q_kink is: {q_l}", q_l= q_l)
    #print("Mean Utility is: ", jnp.mean(sim_v_result))
    jax.debug.print("Mean Revenue is: {x}", x= jnp.mean(sim_r_result))
    return sim_result

cf_ps_change_ql_jitted = jax.jit(cf_ps_change_ql)

def cf_ps_change_q (q):
    q_kink_l = jnp.array([2.0, 6.0, 11.0, 20.0])
    target_index = find_target_index_q_jitted(q)
    q_l = q_kink_l.at[target_index].set(q)
    return cf_ps_change_ql_jitted(q_l)

cf_ps_change_q_jitted = jax.jit(cf_ps_change_q)

q_sequence = jnp.array(jnp.arange(0, 20.1, 0.1))

#cs_p_change_df = cf_cs_change_p(3.09)
#cs_p_change_df_base = cs_p_change_df.groupby(['prem_id'])['sim_result_v'].mean().reset_index()

ps_q_change = jnp.array([cf_ps_change_q_jitted(x) for x in q_sequence])

#cs_p_change = jnp.apply_along_axis(cf_cs_change_p_jitted, axis=1, arr = p_sequence)

ps_q_change_stacked = jnp.column_stack(ps_q_change)

column_names = [f'{i:.5f}' for i in q_sequence for _ in range(2)]

column_names = np.array([f'{float(s.split("_")[0]):.5f}_q' if idx % 2 == 0 else f'{float(s.split("_")[0]):.5f}_r' for idx, s in enumerate(column_names)])
ps_q_change_df = pd.DataFrame(ps_q_change_stacked, columns = column_names)

ps_q_change_df.loc[:, 'prem_id'] = demand_2018_using['prem_id']
ps_q_change_df.loc[:, 'bill_ym'] = demand_2018_using['bill_ym']

ps_q_change_df.to_csv('cf_ps_q_change.csv', index=False)

ps_q_change_df_mean_premid = ps_q_change_df.groupby('prem_id').agg({col: 'mean' for col in ps_q_change_df.columns if col not in ['prem_id', 'bill_ym']})
ps_q_change_df_mean_premid_long = pd.melt(ps_q_change_df_mean_premid.reset_index(), id_vars=['prem_id'], var_name='variable', value_name='value')
ps_q_change_df_mean_premid_long[['quantity', 'type']] = ps_q_change_df_mean_premid_long['variable'].str.split('_', expand=True)
ps_q_change_df_mean_premid_long = ps_q_change_df_mean_premid_long.drop('variable', axis=1)
ps_q_change_df_mean_premid_long = ps_q_change_df_mean_premid_long[['prem_id', 'quantity', 'type', 'value']]
ps_q_change_df_mean_premid_long.to_csv('ps_q_change_mean_premid.csv', index=False)

ps_q_change_df_mean_billym = ps_q_change_df.groupby('bill_ym').agg({col: 'mean' for col in ps_q_change_df.columns if col not in ['prem_id', 'bill_ym']})
ps_q_change_df_mean_billym_long = pd.melt(ps_q_change_df_mean_billym.reset_index(), id_vars=['bill_ym'], var_name='variable', value_name='value')
ps_q_change_df_mean_billym_long[['quantity', 'type']] = ps_q_change_df_mean_billym_long['variable'].str.split('_', expand=True)
ps_q_change_df_mean_billym_long = ps_q_change_df_mean_billym_long.drop('variable', axis=1)
ps_q_change_df_mean_billym_long = ps_q_change_df_mean_billym_long[['bill_ym', 'quantity', 'type', 'value']]
ps_q_change_df_mean_billym_long.to_csv('ps_q_change_mean_billym.csv', index=False)