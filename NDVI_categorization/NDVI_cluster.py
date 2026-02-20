import os
os.environ['JAX_ENABLE_X64'] = 'true'
import numpy as np
import pandas as pd
import statsmodels.api as sm
#from pandas.stats.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.optimize import minimize, fsolve
from jax.scipy.special import erf
import gc
import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, jit, lax
#import cobyqa
#from cobyqa import minimize
#from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
#from scipy.optimize import BFGS
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


demand_2018_using_eta = pd.read_csv('../demand_2018_using_eta.csv')

demand_2018_using_new = pd.read_csv('../demand_2018_using_new.csv')

demand_2018_using_eta = demand_2018_using_eta[demand_2018_using_eta['bill_ym'] >= 201901]
demand_2018_using_new = demand_2018_using_new[demand_2018_using_new['bill_ym'] >= 201901]

demand_2018_using_eta['income'] = demand_2018_using_new['income']
demand_2018_using_eta['charge'] = demand_2018_using_new['charge']

demand_2018_using_new = demand_2018_using_new[demand_2018_using_new['charge'] <= demand_2018_using_new['income']]
demand_2018_using_eta = demand_2018_using_eta[demand_2018_using_eta['charge'] <= demand_2018_using_eta['income']]

demand_2018_using_new = demand_2018_using_new[demand_2018_using_new['income']>1200]
demand_2018_using_eta = demand_2018_using_eta[demand_2018_using_eta['income'] >1200]

ids_to_exclude = [956860320, 3672440428, 8396470324]

# Filter the DataFrame to keep only rows where 'prem_id' is NOT in the list
demand_2018_using_new = demand_2018_using_new[~demand_2018_using_new['prem_id'].isin(ids_to_exclude)]
demand_2018_using_eta = demand_2018_using_eta[~demand_2018_using_eta['prem_id'].isin(ids_to_exclude)]


p_l0 = jnp.array([3.09,  5.01,  8.54, 12.9 , 14.41])
q_l0 = jnp.array([2, 6, 11, 20])
fc_l0 = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.25+29.75])
p_l0_CAP =  jnp.array([2.37+0.05, 4.05+0.05, 6.67+0.05, 11.51+0.05, 14.21+0.05])

result = np.genfromtxt('../result.csv', delimiter=',', skip_header=0)

beta = result[0]
se = result[1]

b1 = jnp.array([beta[0], beta[1]
                #,  beta[2], beta[3], beta[4]
                ])
b2 = jnp.array([beta[2], beta[3], beta[4], beta[5]])
c_o = beta[6]
b4 = jnp.array([beta[7], beta[8], beta[9], beta[10]
                #, beta[14]
                ])

c_alpha = beta[11]
b6 = jnp.array([beta[12], beta[13], beta[14]])
c_rho = beta[15]
sigma_eta = beta[16]
sigma_nu = beta[17]

np.random.seed(42)
sim = 100
#key = random.PRNGKey(42)
len_transactions = len( demand_2018_using_new) 
shape = (len_transactions, sim)
nu_array = np.random.normal(loc = 0, scale = sigma_nu, size = shape)
#eta_l = jnp.array(demand_2018_using_eta['e_diff'])

eta_l = jnp.array(demand_2018_using_eta['mean_e_diff'])

def calculate_log_w(p_l, q_l, fc_l, Z, I):
    p_l_CAP = p_l-p_l0 + p_l0_CAP
    q_kink_l = q_l
    p_plus1_l = jnp.append(p_l[1:5],jnp.array([jnp.nan]) )
    d_end = jnp.cumsum( (p_plus1_l-0.2 - (p_l-0.2)) [:4] *q_kink_l)
    d_end =  jnp.insert(d_end, 0, jnp.array([0.0]) )
    
    p_plus1_l_CAP = jnp.append(p_l_CAP[1:5],jnp.array([jnp.nan]) )
    d_end_CAP = jnp.cumsum( (p_plus1_l_CAP-0.2 - (p_l_CAP-0.2)) [:4] *q_kink_l)
    d_end_CAP =  jnp.insert(d_end_CAP, 0, jnp.array([0.0]) )
    
    def calculate_dk (k):
        result = -fc_l[k] + d_end[k]
        return result
    calculate_dk_jitted = jax.jit(calculate_dk)
    
    def calculate_dk_CAP (k):
        result = -fc_l[k] + d_end_CAP[k]
        return result
    calculate_dk_CAP_jitted = jax.jit(calculate_dk_CAP)
    
    def get_total_wk (beta_1, beta_2,
                  c_wo,
                  beta_4, 
                  c_a,
                  beta_6,
                  c_r,
                  k, 
                  Z,
                  A_i = A_current_income,
                  A_p = A_current_price,
                  A_o = A_current_outdoor,
                  G = G,
                  p = p_l, I = I,
                  p0 =p0, 
                  de = de,
                  ):
        p_k = jnp.where(CAP == 1, p_l_CAP[k], p_l[k])
        d_k = jnp.where(CAP == 1, calculate_dk_CAP_jitted(k), calculate_dk_jitted(k))
        alpha = jnp.exp(jnp.dot(A_p, beta_4)
                    + c_a
                    )
        rho = abs(jnp.dot(A_i, beta_6)
                    + c_r
                    )
        w = jnp.exp(jnp.dot(A_o, beta_1) + jnp.dot(Z, beta_2)
                       - jnp.multiply( jnp.multiply(alpha,jnp.log(p_k)), de) + 
                       jnp.multiply(rho, jnp.log(jnp.maximum( I+ jnp.multiply(d_k, de), 1e-16))) + c_wo)
        result = jnp.log(w)
        return result

    get_total_wk_jitted = jax.jit(get_total_wk)

    def get_total_wk_k (k):
        result = get_total_wk_jitted(beta_1 = b1,beta_2 = b2,
                                 c_wo = c_o,beta_4 = b4,c_a = c_alpha,
                       beta_6 = b6, c_r = c_rho,
                       k=k,
                       Z = Z
                       )
        return result

    get_total_wk_k_jitted = jax.jit(get_total_wk_k)

    log_w = jnp.column_stack((get_total_wk_k_jitted(0), get_total_wk_k_jitted(1), get_total_wk_k_jitted(2),
                    get_total_wk_k_jitted(3), get_total_wk_k_jitted(4)))
    return log_w

calculate_log_w_jitted = jax.jit(calculate_log_w)

def gen_nu_array(sigma_mu):
    #shape = (sim, 1)
    shape = (len_transactions, sim)
    #nu_array = sigma_nu * random.normal(key, shape)
    nu_array = np.random.normal(loc = 0, scale = sigma_nu, size = shape)
    nu_array = jnp.minimum(nu_array, 7)
    nu_array = jnp.maximum(nu_array, -7)
    return nu_array
gen_nu_array_jitted = jax.jit(gen_nu_array)

def get_log_q_inner(log_w_k, q_l):
    e = log_w_k[-1]
    log_w_k = log_w_k[:-1]
    log_w1 = log_w_k[0]
    log_w2 = log_w_k[1]
    log_w3 = log_w_k[2]
    log_w4 = log_w_k[3]
    log_w5 = log_w_k[4]
    n = log_w_k[-sim:]
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
    result = result.reshape(1, -1)
    return result
get_log_q_inner_jitted = jax.jit(get_log_q_inner)

def cf_w (p_l, q_l, fc_l, Z, I,
          #nu_array = nu_array,
          eta_l = eta_l):
    log_q_sim = get_log_q_sim_jitted(p_l, q_l, fc_l, Z, I)
    log_q_sim = log_q_sim.reshape(len_transactions, sim)
    gc.collect()
    #nu_array =  gen_nu_array_jitted(sigma_nu)
    return log_q_sim
# + nu_array
cf_w_jitted = jax.jit(cf_w)


def get_log_w(p_l, q_l, fc_l,Z, I,
          nu_array = nu_array,
          eta_l = eta_l):
    log_w = calculate_log_w_jitted(p_l, q_l, fc_l, Z , I)
    #log_w = jnp.column_stack((log_w, eta_l))
    log_w = jnp.column_stack((log_w, nu_array))
    return log_w
get_log_w_jitted = jax.jit(get_log_w)

def get_log_q_sim(p_l, q_l, fc_l, Z, I,
          nu_array = nu_array,
          eta_l = eta_l):
    #nu_array = gen_nu_array_jitted(sigma_nu)
    log_w = get_log_w_jitted(p_l, q_l, fc_l, Z, I)
    log_w = jnp.column_stack((log_w, eta_l))
    
    def get_log_q (log_w_k, q_l = q_l):
        #log_q_nonu = get_log_q_inner_jitted(log_w_k, n, q_l)
        #nu_array = gen_nu_array_jitted(sigma_nu)
        return get_log_q_inner_jitted(log_w_k,q_l)
    get_log_q_jitted = jax.jit(get_log_q)
    
    log_q_sim = jnp.apply_along_axis(get_log_q_jitted, axis=1, arr = log_w)
    #nu_array = gen_nu_array_jitted(sigma_nu)
    return log_q_sim
get_log_q_sim_jitted = jax.jit(get_log_q_sim)


A_current_outdoor = jnp.column_stack(( 
    jnp.array(demand_2018_using_new['bathroom']), 
                                      #jnp.array( demand_2018_using_new['prev_NDVI']),
                                      jnp.zeros_like(demand_2018_using_new['prev_NDVI']),
                                      ))
A_current_indoor = jnp.column_stack((jnp.array(demand_2018_using_new['bathroom']),
                                     jnp.array(demand_2018_using_new['above_one_acre'])
                                       ))
A_current_price = jnp.column_stack((
    jnp.array(demand_2018_using_new['bedroom']), 
    #jnp.array( demand_2018_using_new['prev_NDVI']),
    jnp.zeros_like(demand_2018_using_new['prev_NDVI']), 
    jnp.array(demand_2018_using_new['mean_TMAX_1']),
    jnp.array(demand_2018_using_new['total_PRCP'])
    ))

A_current_income = jnp.column_stack((
    jnp.array(demand_2018_using_new['heavy_water_app']), 
    jnp.array(demand_2018_using_new['bedroom']), 
    #jnp.array( demand_2018_using_new['prev_NDVI']),
    jnp.zeros_like(demand_2018_using_new['prev_NDVI']), 
    ))

Z_current_outdoor = jnp.column_stack((jnp.array(demand_2018_using_new['mean_TMAX_1']),
                                      jnp.array(demand_2018_using_new['IQR_TMAX_1']),
                                      jnp.array(demand_2018_using_new['total_PRCP']) 
                                      ,jnp.array(demand_2018_using_new['IQR_PRCP'])
                                      ))
Z_current_indoor = jnp.array(demand_2018_using_new['mean_TMAX_1'])
Z_current_indoor = Z_current_indoor[:, jnp.newaxis]
#G =jnp.array( demand_2018_using_new['prev_NDVI'])
G = jnp.zeros_like(demand_2018_using_new['prev_NDVI'])
I = jnp.array(demand_2018_using_new['income'])
p0 = jnp.array(demand_2018_using_new['previous_essential_usage_mp'])
w_i = jnp.array(demand_2018_using_new['quantity'])
de = jnp.array(demand_2018_using_new['deflator'])
q_statusquo = jnp.array(demand_2018_using_new['quantity'])

q_statusquo_sum = jnp.sum(q_statusquo)

alpha = jnp.exp(jnp.dot(A_current_price, b4)
                    + c_alpha)

rho = jnp.exp(jnp.dot(A_current_income, b6)
                    + c_rho)

demand_2018_using_new.loc[:, 'e_alpha'] = alpha
demand_2018_using_new.loc[:, 'e_rho'] = rho

CAP = jnp.array(demand_2018_using_new['CAP_HH'])

Z_current = Z_current_outdoor

I_current = I

log_q0 = cf_w_jitted(p_l0, q_l0, fc_l0, Z_current, I_current)

q0 = jnp.exp(log_q0)

def nansum_ignore_nan_inf(arr):
    mask = jnp.logical_and(jnp.isfinite(arr), ~jnp.isnan(arr))  # Mask out inf and NaN
    return jnp.sum(jnp.where(mask, arr, 0), axis = 0)
nansum_ignore_nan_inf_jitted = jax.jit(nansum_ignore_nan_inf)

def sum_ignore_outliers(arr, lower_percentile=25, upper_percentile=75):
    # Compute the IQR along axis=0
    q1 = jnp.percentile(arr, lower_percentile, axis=0)
    q3 = jnp.percentile(arr, upper_percentile, axis=0)
    iqr = q3 - q1

    # Define outlier boundaries
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Apply condition using lax.select instead of boolean indexing
    in_bounds = (arr >= lower_bound) & (arr <= upper_bound)

    # Use jax.lax.select to retain values within bounds, and set out-of-bounds values to 0
    filtered_arr = lax.select(in_bounds, arr, jnp.zeros_like(arr))

    # Compute the sum of the filtered array along axis=0
    return jnp.sum(filtered_arr, axis=0)

# JIT the function
sum_ignore_outliers_jitted = jax.jit(sum_ignore_outliers)

q0_sum =nansum_ignore_nan_inf_jitted(q0)

def nanmean_ignore_nan_inf(arr):
    mask = jnp.logical_and(jnp.isfinite(arr), ~jnp.isnan(arr))  # Mask out inf and NaN
    return jnp.mean(jnp.where(mask, arr, 0), axis =1)
nanmean_ignore_nan_inf_jitted = jax.jit(nanmean_ignore_nan_inf)

#q0_sum =nansum_ignore_nan_inf_jitted(q0)
q0_mean = nanmean_ignore_nan_inf_jitted(q0)

def get_r_mean(q_mean, p_l, q_l, fc_l):
    def expenditure_func(w, p_l=p_l, q_l=q_l, fc_l=fc_l):
        bins = jnp.concatenate((jnp.array([0]), q_l, jnp.array([jnp.inf])))
        binned_data = jnp.digitize(w, bins)
        q_plus1_l = jnp.insert(q_l, 0, 0)
        q_diff_l = q_l - q_plus1_l[0:4]
        cumu_sum = jnp.cumsum(p_l[0:4] * q_diff_l)
        result = jnp.where(binned_data==1, fc_l[0] + p_l[0]*w, 
                           fc_l[binned_data-1] + cumu_sum[binned_data-2] + p_l[binned_data-1] * (w - q_l[binned_data-2]))
        return result
    expenditure_func_jitted = jax.jit(expenditure_func)
    r_mean = expenditure_func_jitted(q_mean)
    return r_mean
get_r_mean_jitted = jax.jit(get_r_mean)

hh_size = len(np.unique(np.array(demand_2018_using_new['prem_id'], dtype = np.int64)))

def parsing():
    pass

def nansum_ignore_nan_inf_otheraxis(arr):
    mask = jnp.logical_and(jnp.isfinite(arr), ~jnp.isnan(arr))  # Mask out inf and NaN
    return jnp.sum(jnp.where(mask, arr, 0), axis = 1)
nansum_ignore_nan_inf_otheraxis_jitted = jax.jit(nansum_ignore_nan_inf_otheraxis)

def sum_ignore_outliers_otheraxis(arr, lower_percentile=25, upper_percentile=75):
    # Compute the IQR along axis=0
    q1 = jnp.percentile(arr, lower_percentile, axis=1)
    q3 = jnp.percentile(arr, upper_percentile, axis=1)
    iqr = q3 - q1

    # Define outlier boundaries
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Apply condition using lax.select instead of boolean indexing
    in_bounds = (arr >= lower_bound) & (arr <= upper_bound)

    # Use jax.lax.select to retain values within bounds, and set out-of-bounds values to 0
    filtered_arr = lax.select(in_bounds, arr, jnp.zeros_like(arr))

    # Compute the sum of the filtered array along axis=1
    return jnp.sum(filtered_arr, axis=1)

# JIT the function
sum_ignore_outliers_otheraxis_jitted = jax.jit(sum_ignore_outliers_otheraxis)

def get_q_sum(p_l, q_l, fc_l, Z, I):
    log_q = cf_w_jitted(p_l, q_l, fc_l, Z, I)
    chunk_size = 100 
    num_columns = log_q.shape[1]
    result_hh = []
    result_sim = []
    for start_col in range(0, num_columns, chunk_size):
        end_col = min(start_col + chunk_size, num_columns)
        data_chunk = log_q[:, start_col:end_col]
        result_chunk = jnp.exp(data_chunk)
        result_chunk_hh = nansum_ignore_nan_inf_otheraxis_jitted(result_chunk)
        result_hh.append(result_chunk_hh)
        result_chunk_sim = nansum_ignore_nan_inf_jitted(result_chunk)
        result_sim.append(result_chunk_sim)

    q_sum_hh = jnp.transpose(jnp.vstack(result_hh))
    q_sum_hh = jnp.sum(q_sum_hh, axis = 1)
    q_sum_sim = jnp.concatenate(result_sim, axis = 0)
    #q = jnp.exp(log_q)
    #q_sum = nansum_ignore_nan_inf_jitted(q)
    return q_sum_hh, q_sum_sim
get_q_sum_jitted = jax.jit(get_q_sum)

def get_q_sum_hh(p_l, q_l, fc_l, Z, I):
    log_q = cf_w_jitted(p_l, q_l, fc_l, Z, I)
    
    #chunk_size = 100 
    #num_columns = log_q.shape[1]
    #result_hh = []
    #result_sim = []
    #for start_col in range(0, num_columns, chunk_size):
     #   end_col = min(start_col + chunk_size, num_columns)
      #  data_chunk = log_q[:, start_col:end_col]
      #  result_chunk = jnp.exp(data_chunk)
        #result_chunk_hh = nansum_ignore_nan_inf_otheraxis_jitted(result_chunk)
       # result_hh.append(result_chunk)
        #result_chunk_sim = nansum_ignore_nan_inf_jitted(result_chunk)
        #result_sim.append(result_chunk_sim)

    #q_sum_hh = jnp.transpose(jnp.vstack(result_hh))
    q = jnp.exp(log_q)
    q_sum_hh = nansum_ignore_nan_inf_otheraxis_jitted(q)
    #q_sum_hh = jnp.sum(q_sum_hh, axis = 1)
    #q_sum_sim = jnp.concatenate(result_sim, axis = 0)
    #q = jnp.exp(log_q)
    #q_sum = nansum_ignore_nan_inf_jitted(q)
    return q_sum_hh
get_q_sum_hh_jitted = jax.jit(get_q_sum_hh)

def get_q_sum_sim(p_l, q_l, fc_l, Z, I):
    log_q = cf_w_jitted(p_l, q_l, fc_l, Z, I)
    #chunk_size = 100 
    #num_columns = log_q.shape[1]
    #result_hh = []
    #result_sim = []
    #for start_col in range(0, num_columns, chunk_size):
     #   end_col = min(start_col + chunk_size, num_columns)
      #  data_chunk = log_q[:, start_col:end_col]
       # result_chunk = jnp.exp(data_chunk)
        #result_chunk_hh = nansum_ignore_nan_inf_otheraxis_jitted(result_chunk)
        #result_hh.append(result_chunk_hh)
        #result_chunk_sim = nansum_ignore_nan_inf_jitted(result_chunk)
        #result_sim.append(result_chunk)

    #q_sum_hh = jnp.transpose(jnp.vstack(result_hh))
    #q_sum_hh = jnp.sum(q_sum_hh, axis = 1)
    q = jnp.exp(log_q)
    q_sum_sim = sum_ignore_outliers_jitted(q)
    #q = jnp.exp(log_q)
    #q_sum = nansum_ignore_nan_inf_jitted(q)
    return q_sum_sim
get_q_sum_sim_jitted = jax.jit(get_q_sum_sim)

######################
#### Revenue #####
########################

def from_q_to_r_mean(q_sum_hh, p_l, q_l, fc_l):
    q_mean = q_sum_hh / sim
    r_mean = get_r_mean_jitted(q_mean, p_l, q_l, fc_l)
    return r_mean
from_q_to_r_mean_jitted = jax.jit(from_q_to_r_mean)

def from_q_to_r(q_sum_hh, p_l, q_l, fc_l):
    r_mean = from_q_to_r_mean_jitted(q_sum_hh, p_l, q_l, fc_l)
    #r = get_r_jitted(r_mean)
    return r_mean
from_q_to_r_jitted = jax.jit(from_q_to_r)

def get_r(r_mean, prem_id = None):
    if not prem_id:
        prem_id =  jnp.array(demand_2018_using_new['prem_id'], dtype = jnp.int64)
    
    id_rmean = jnp.transpose(jnp.vstack((prem_id, r_mean)))
    groups = id_rmean[:,0].copy()
    id_rmean = jnp.delete(id_rmean, 0, axis=1)
    _ndx = jnp.argsort(groups)
    _id, _pos, g_count  = jnp.unique(groups[_ndx], 
                                return_index=True, 
                                return_counts=True, size=hh_size)
    g_sum = jnp.ufunc(jnp.add, 2, 1).reduceat(id_rmean[_ndx], _pos, axis=0)
    r = g_sum / g_count[:,None]
    return r

get_r_jitted = jax.jit(get_r)

q_sum_hh0 = get_q_sum_hh_jitted(p_l0, q_l0, fc_l0, Z_current, I_current)
r0 = from_q_to_r_jitted(q_sum_hh0, p_l0, q_l0, fc_l0)

#del A_current_indoor, demand_2018_using_new, demand_2018_using_eta, w_i

######################
#### NDVI Clustering #####
########################
import os
# This forces the library to skip the complex thread detection that is crashing
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"

# --- SECTION 1: DATA PREPARATION ---

# 1. Create the working DataFrame
# Assuming 'demand_2018_using_new' is your raw dataframe loaded in memory
NDVI_df = demand_2018_using_new[['bill_ym', 'prem_id', 'NDVI', 'mean_TMAX_1', 'IQR_TMAX_1', 'total_PRCP', 'IQR_PRCP']]
df = NDVI_df.copy()

# 2. Rank Transform (Normalize Units)
# We use percentiles so Temperature (deg F) and Precip (inches) are comparable
for v in ['mean_TMAX_1', 'IQR_TMAX_1', 'total_PRCP', 'IQR_PRCP']:
    df[v + '_r'] = df[v].rank(pct=True)

# 3. Create Weather Indices
# Weather Level: High = Wet/Cool (Winter), Low = Hot/Dry (Summer)
df['weather_level'] = df['total_PRCP_r'] - df['mean_TMAX_1_r']

# Weather Volatility: High = Stormy/Unstable, Low = Consistent
df['weather_vol'] = df['IQR_PRCP_r'] + df['IQR_TMAX_1_r']

# Create numeric month for seasonality analysis
df['month'] = df['bill_ym'] % 100

# --- SECTION 2: FEATURE ENGINEERING (AGGREGATION) ---

print("Calculating household features... (This may take a moment)")

# 1. Basic Stats (Mean & Std of Greenness)
hh_features = df.groupby('prem_id').agg(
    mean_ndvi=('NDVI', 'mean'),
    std_ndvi=('NDVI', 'std')
)

# 2. Seasonal Logic
# Summer = June, July, Aug; Winter = Dec, Jan, Feb
summer_means = df[df['month'].isin([6, 7, 8])].groupby('prem_id')['NDVI'].mean()
winter_means = df[df['month'].isin([1, 2, 12])].groupby('prem_id')['NDVI'].mean()
hh_features['season_gap'] = summer_means - winter_means

# 3. Correlation (Weather Sensitivity)
# This measures if the lawn follows the sun (Negative Corr) or needs rain (Positive Corr)
def safe_corr_grouped(g):
    if len(g) < 6:
        return np.nan
    x = g['NDVI'].values
    y = g['weather_level'].values
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return np.corrcoef(x, y)[0, 1]

# Note: include_groups=False fixes the Pandas 2.2+ warning
hh_features['corr_level'] = df.groupby('prem_id')[['NDVI', 'weather_level']].apply(
    safe_corr_grouped, 
    include_groups=False
)

# Fill NaNs (e.g., properties with missing seasonal data)
hh_features = hh_features.fillna(0)

# --- SECTION 3: CLUSTERING ---

# 1. Scale the Data
# Correlation is -1 to 1; NDVI is 0 to 1. Scaling makes them equal weight.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(hh_features[['mean_ndvi', 'corr_level', 'season_gap']])

# 2. Run K-Means (4 Clusters for our 4 Archetypes)
# TO THIS (adding algorithm='elkan' sometimes bypasses the specific Lloyd implementation bug):
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, algorithm='elkan')
hh_features['cluster'] = kmeans.fit_predict(X_scaled)

# --- SECTION 4: AUTO-LABELING ---

# We identify the clusters based on their Centroids (Averages)
centroids = hh_features.groupby('cluster')[['mean_ndvi', 'corr_level']].mean()

# --- RE-RUNNING LABELING WITH TUNED THRESHOLDS ---

def get_label(row):
    # 1. Catch the Rain Driven (Cluster 2)
    if row['corr_level'] > 0.10:
        return "B: Rain Driven (Water Constrained)"
        
    # 2. Catch the Strong Sun Lovers (Cluster 3)
    if row['corr_level'] < -0.25:
        return "A: Sun Lover (Temp Driven)"

    # 3. Split the Weak Responders (Clusters 0 and 1)
    if row['mean_ndvi'] > 0.40:  # Cluster 0 is 0.46, so 0.40 is safe
        return "D: Irrigator (Artificial)"
    else:
        return "C: Concrete (Dead/Paved)"

# Re-map the labels
cluster_map = {c: get_label(centroids.loc[c]) for c in centroids.index}
hh_features['label'] = hh_features['cluster'].map(cluster_map)

# Print the new counts
print("New Cluster Counts:")
print(hh_features['label'].value_counts())

# --- SECTION 5: VISULIZATION ---

# --- STEP 0: SETUP OUTPUT DIRECTORY ---
output_dir = 'plot'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- STEP 1: PREPARE DATA ---
if 'label' in df.columns:
    df = df.drop(columns=['label'])
df_merged = df.merge(hh_features[['label']], left_on='prem_id', right_index=True)

# Define the strict order for the legend
label_order = sorted(hh_features['label'].unique())

# --- PLOT 1: COMPACT DOT PLOT (SORTED LEGEND) ---
plt.figure(figsize=(7, 5)) 

sns.scatterplot(
    data=hh_features, 
    x='corr_level', 
    y='mean_ndvi', 
    hue='label', 
    hue_order=label_order, # <--- THIS FORCES A, B, C, D ORDER
    style='label', 
    style_order=label_order,
    palette='deep', 
    s=30, 
    alpha=0.6
)

plt.title('Segmentation Map', fontsize=12)
plt.xlabel('Correlation', fontsize=10)
plt.ylabel('Mean NDVI', fontsize=10)
plt.axvline(0, color='black', linestyle='--', alpha=0.3)
plt.axhline(0.40, color='black', linestyle='--', alpha=0.3)

# Move legend outside and keep it compact
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title='Cluster Type')

plt.tight_layout()
save_path_1 = f"{output_dir}/segmentation_dot_plot_sorted.png"
plt.savefig(save_path_1, dpi=100)
plt.close()
print(f"Saved Plot 1 (Sorted Legend) to: {save_path_1}")


# --- PLOT 2: COMPACT 4-PANEL DASHBOARD ---

# Aggregate monthly profiles
cluster_profiles = df_merged.groupby(['label', 'month'])[['NDVI', 'mean_TMAX_1', 'total_PRCP']].mean().reset_index()

# Colors
color_ndvi = 'forestgreen'
color_temp = 'firebrick'
color_prcp = 'dodgerblue'

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Iterate through the SORTED labels so the panels are also A, B, C, D
for i, label in enumerate(label_order):
    ax = axes[i]
    data = cluster_profiles[cluster_profiles['label'] == label]
    
    # --- Axis 1: Precipitation (Bars) ---
    ax_prcp = ax.twinx()
    ax_prcp.bar(data['month'], data['total_PRCP'], color=color_prcp, alpha=0.2)
    ax_prcp.set_ylabel('Precip (Inches)', color=color_prcp, fontsize=9)
    ax_prcp.tick_params(axis='y', labelsize=8)
    ax_prcp.set_ylim(0, 6)
    
    # --- Axis 2: Temperature (Red Line) ---
    ax_temp = ax.twinx()
    ax_temp.spines["right"].set_position(("axes", 1.15)) 
    
    ax_temp.plot(data['month'], data['mean_TMAX_1'], color=color_temp, linestyle='--', linewidth=1.5)
    ax_temp.set_ylabel('Temp (F)', color=color_temp, fontsize=9)
    ax_temp.tick_params(axis='y', labelsize=8)
    ax_temp.set_ylim(30, 110)

    # --- Axis 3: NDVI (Green Line) ---
    ax.plot(data['month'], data['NDVI'], color=color_ndvi, linewidth=2.5, marker='o', markersize=4)
    
    # Formatting
    short_label = label.split('(')[0].strip() 
    ax.set_title(short_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('NDVI', color=color_ndvi, fontsize=10, fontweight='bold')
    ax.tick_params(axis='y', labelsize=9)
    ax.set_ylim(0.1, 0.8)
    ax.set_xticks(range(1, 13, 2))
    ax.set_xlabel('') 
    ax.grid(True, alpha=0.3)
    
    # Clean up grid
    ax_prcp.grid(False)
    ax_temp.grid(False)

# Layout adjustments
plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.85, wspace=0.4, hspace=0.3)

save_path_2 = f"{output_dir}/cluster_description.png"
plt.savefig(save_path_2, dpi=100)
plt.close()
print(f"Saved Plot 2 to: {save_path_2}")

# --- STEP 0: SETUP OUTPUT DIRECTORY ---
output_dir = '../' # or change to 'data' or wherever you prefer
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- STEP 1: SELECT THE COLUMNS YOU NEED ---
# We use 'hh_features' because it is already at the Household level (unique prem_id)
# We include the label, the cluster ID, and the key metrics (NDVI, Correlation)
output_df = hh_features[['label', 'mean_ndvi', 'corr_level', 'season_gap', 'cluster']].copy()

# --- STEP 2: CLEAN UP THE FORMATTING ---
# Sort by Label so the CSV is organized (A, B, C, D)
output_df = output_df.sort_values(by=['label', 'mean_ndvi'], ascending=[True, False])

# Rename index to be clear (it's currently the index, let's make it a column)
output_df.index.name = 'prem_id'

# --- STEP 3: SAVE TO CSV ---
save_path = f"{output_dir}/premise_segments_roster.csv"
output_df.to_csv(save_path)

print(f"Successfully saved categorization for {len(output_df)} premises.")
print(f"File saved to: {save_path}")
print("\nPreview of the file:")
print(output_df.head())