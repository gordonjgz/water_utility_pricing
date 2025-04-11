import os
os.environ['JAX_ENABLE_X64'] = 'true'
import numpy as np
import pandas as pd
import statsmodels.api as sm
#from pandas.stats.api import ols
import matplotlib.pyplot as plt
#import seaborn as sns
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


demand_2018_using_eta = pd.read_csv('../demand_2018_using_eta.csv')

demand_2018_using_new = pd.read_csv('../demand_2018_using_new.csv')

demand_2018_using_eta = demand_2018_using_eta[demand_2018_using_eta['bill_ym'] >= 201901]
demand_2018_using_new = demand_2018_using_new[demand_2018_using_new['bill_ym'] >= 201901]


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
eta_l = jnp.array(demand_2018_using_eta['e_diff'])

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
                                      jnp.array( demand_2018_using_new['prev_NDVI']),
                                      ))
A_current_indoor = jnp.column_stack((jnp.array(demand_2018_using_new['bathroom']),
                                     jnp.array(demand_2018_using_new['above_one_acre'])
                                       ))
A_current_price = jnp.column_stack((
    jnp.array(demand_2018_using_new['bedroom']), 
    jnp.array(demand_2018_using_new['prev_NDVI']), 
    jnp.array(demand_2018_using_new['mean_TMAX_1']),
    jnp.array(demand_2018_using_new['total_PRCP'])
    ))

A_current_income = jnp.column_stack((
    jnp.array(demand_2018_using_new['heavy_water_app']), 
    jnp.array(demand_2018_using_new['bedroom']), 
    jnp.array(demand_2018_using_new['prev_NDVI']), 
    ))

Z_current_outdoor = jnp.column_stack((jnp.array(demand_2018_using_new['mean_TMAX_1']),
                                      jnp.array(demand_2018_using_new['IQR_TMAX_1']),
                                      jnp.array(demand_2018_using_new['total_PRCP']) 
                                      ,jnp.array(demand_2018_using_new['IQR_PRCP'])
                                      ))
Z_current_indoor = jnp.array(demand_2018_using_new['mean_TMAX_1'])
Z_current_indoor = Z_current_indoor[:, jnp.newaxis]
G = jnp.array(demand_2018_using_new['prev_NDVI'])
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

#### Calculating Price Elasticity

p_plus = p_l0*1.05
fc_plus = fc_l0*1.05

eq_0 = q_sum_hh0/sim

q_sum_hh_plus = get_q_sum_hh_jitted(p_plus, q_l0, fc_l0, Z_current, I_current)

eq_plus = q_sum_hh_plus/sim

price_elasticity = jnp.divide((eq_plus - eq_0), 0.05*eq_0)
#This \price elasticity" is the percent change in household i's expected water consumption as a result of
# a 5 percent increase in all prices on the nonlinear price function divided 0.05.

price_elasticity_df = demand_2018_using_new[['prem_id', 'bill_ym', 'income', 'heavy_water_spa', 'prev_NDVI', 'bedroom', 'bathroom']].copy()
price_elasticity = np.array(price_elasticity)
np.mean(price_elasticity)
# -0.4492176148432877
np.median(price_elasticity)
# -0.4209277545524498
# Create new DataFrame
price_elasticity_df.loc[:, 'price_elasticity'] = price_elasticity

summary_stats = price_elasticity_df.groupby('prem_id').agg(
    mean_price_elasticity=('price_elasticity', 'mean'),
    median_price_elasticity=('price_elasticity', 'median'),
    mean_income=('income', 'mean'),
    median_income=('income', 'median'),
    heavy_water_spa=('heavy_water_spa', 'mean'),
    bedroom=('bedroom', 'mean'),
    bathroom = ('bathroom', 'mean'),
    NDVI=('prev_NDVI', 'mean')
).reset_index()


plt.figure(figsize=(8, 6))
plt.scatter(summary_stats['mean_income'], summary_stats['median_price_elasticity'], alpha=0.5)
plt.title('Income vs. Price Elasticity', fontsize=20)
plt.xlabel('Avg Monthly HH Income ($)', fontsize=20)
plt.ylabel('Price Elasticity', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)  # Adjust major ticks
plt.tick_params(axis='both', which='minor', labelsize=20)  # Adjust minor ticks (optional)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(pad=4.0)  # Adds padding around titles/labels
plt.savefig('pics/price_elasticity_income.png',bbox_inches='tight')

plt.figure(figsize=(8, 6))
plt.scatter(summary_stats['heavy_water_spa'], summary_stats['median_price_elasticity'], alpha=0.5)
plt.title('Heavy Water Spa vs. Price Elasticity', fontsize=20)
plt.xlabel('Heavy Water or Spa Appliances', fontsize=20)
plt.ylabel('Price Elasticity', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)  # Adjust major ticks
plt.tick_params(axis='both', which='minor', labelsize=20)  # Adjust minor ticks (optional)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(pad=4.0)  # Adds padding around titles/labels
plt.savefig('pics/price_elasticity_heavy_water_spa.png', bbox_inches='tight')

plt.figure(figsize=(8, 6))
plt.scatter(summary_stats['bedroom'], summary_stats['median_price_elasticity'], alpha=0.5)
plt.title('Bedroom vs. Price Elasticity', fontsize=20)
plt.xlabel('Bedroom', fontsize=20)
plt.ylabel('Price Elasticity', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)  # Adjust major ticks
plt.tick_params(axis='both', which='minor', labelsize=20)  # Adjust minor ticks (optional)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(pad=4.0)  # Adds padding around titles/labels
plt.savefig('pics/price_elasticity_bedroom.png', bbox_inches='tight')

plt.figure(figsize=(8, 6))
plt.scatter(summary_stats['bathroom'], summary_stats['median_price_elasticity'], alpha=0.5)
plt.title('Bathroom vs. Price Elasticity', fontsize=20)
plt.xlabel('Bathroom', fontsize=20)
plt.ylabel('Price Elasticity', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)  # Adjust major ticks
plt.tick_params(axis='both', which='minor', labelsize=20)  # Adjust minor ticks (optional)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(pad=4.0)  # Adds padding around titles/labels
plt.savefig('pics/price_elasticity_bathroom.png', bbox_inches='tight')

plt.figure(figsize=(8, 6))
plt.scatter(summary_stats['NDVI'], summary_stats['median_price_elasticity'], alpha=0.5)
plt.title('NDVI vs. Price Elasticity', fontsize=20)
plt.xlabel('NDVI', fontsize=20)
plt.ylabel('Price Elasticity', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)  # Adjust major ticks
plt.tick_params(axis='both', which='minor', labelsize=20)  # Adjust minor ticks (optional)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(pad=4.0)  # Adds padding around titles/labels
plt.show()

# Create two groups based on the condition
group_1 = summary_stats[summary_stats['NDVI'] >= 0.39]
group_2 = summary_stats[summary_stats['NDVI'] < 0.39]

# Set up the histogram plot
plt.figure(figsize=(8, 6))

plt.hist(group_1['median_price_elasticity'], bins=60, alpha=0.4, label='NDVI> 0.39', color='blue', edgecolor='black')

plt.hist(group_2['median_price_elasticity'], bins=60, alpha=0.4, label='NDVI < 0.39', color='red', edgecolor='black')

# Add labels and title
plt.xlabel('Price Elasticity', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)  # Adjust major ticks
plt.tick_params(axis='both', which='minor', labelsize=20)  # Adjust minor ticks (optional)
plt.title('Price Elasticity by NDVI', fontsize=20)
plt.legend()
#plt.show()
plt.tight_layout(pad=4.0)  # Adds padding around titles/labels
plt.savefig('pics/price_elasticity_histgram_by_NDVI.png', bbox_inches='tight')

#### Calculating Income Elasticity

I_plus = I_current*1.05

q_sum_hh_plus= get_q_sum_hh_jitted(p_l0, q_l0, fc_l0, Z_current, I_plus)

eq_plus = q_sum_hh_plus/sim

income_elasticity = jnp.divide((eq_plus - eq_0), 0.05*eq_0 )

income_elasticity_df = demand_2018_using_new[['prem_id', 'bill_ym', 'income', 'heavy_water_spa', 'prev_NDVI', 'bedroom', 'bathroom']].copy()
income_elasticity = np.array(income_elasticity)
np.mean(income_elasticity)
#0.10262728881059024
np.median(income_elasticity)
#0.1133137614895198
# Create new DataFrame
income_elasticity_df.loc[:, 'income_elasticity'] = income_elasticity

summary_stats = income_elasticity_df.groupby('prem_id').agg(
    mean_income_elasticity=('income_elasticity', 'mean'),
    median_income_elasticity=('income_elasticity', 'median'),
    mean_income=('income', 'mean'),
    median_income=('income', 'median'),
    heavy_water_spa=('heavy_water_spa', 'mean'),
    bedroom=('bedroom', 'mean'),
    bathroom = ('bathroom', 'mean'),
    NDVI=('prev_NDVI', 'mean')
).reset_index()

plt.figure(figsize=(8, 6))
plt.scatter(summary_stats['mean_income'], summary_stats['median_income_elasticity'], alpha=0.5)
plt.title('Income vs. Income Elasticity', fontsize=20)
plt.xlabel('Avg Monthly HH Income ($)', fontsize=20)
plt.ylabel('Income Elasticity', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)  # Adjust major ticks
plt.tick_params(axis='both', which='minor', labelsize=20)  # Adjust minor ticks (optional)
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig('pics/income_elasticity_income.png',  bbox_inches='tight')

plt.figure(figsize=(8, 6))
plt.scatter(summary_stats['heavy_water_spa'], summary_stats['median_income_elasticity'], alpha=0.5)
plt.title('Heavy Water Spa vs. Income Elasticity', fontsize=20)
plt.xlabel('Heavy Water or Spa Appliances', fontsize=20)
plt.ylabel('Income Elasticity', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)  # Adjust major ticks
plt.tick_params(axis='both', which='minor', labelsize=20)  # Adjust minor ticks (optional)
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig('pics/income_elasticity_heavy_water_spa.png',  bbox_inches='tight')

plt.figure(figsize=(8, 6))
plt.scatter(summary_stats['bedroom'], summary_stats['median_income_elasticity'], alpha=0.5)
plt.title('Bedroom vs. Income Elasticity', fontsize=20)
plt.xlabel('Bedroom', fontsize=20)
plt.ylabel('Income Elasticity', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)  # Adjust major ticks
plt.tick_params(axis='both', which='minor', labelsize=20)  # Adjust minor ticks (optional)
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig('pics/income_elasticity_bedroom.png', bbox_inches='tight')

plt.figure(figsize=(8, 6))
plt.scatter(summary_stats['bathroom'], summary_stats['median_income_elasticity'], alpha=0.5)
plt.title('Bathroom vs. Income Elasticity', fontsize=20)
plt.xlabel('Bathroom', fontsize=20)
plt.ylabel('Income Elasticity', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)  # Adjust major ticks
plt.tick_params(axis='both', which='minor', labelsize=20)  # Adjust minor ticks (optional)
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig('pics/income_elasticity_bathroom.png', bbox_inches='tight')

plt.figure(figsize=(8, 6))
plt.scatter(summary_stats['NDVI'], summary_stats['median_income_elasticity'], alpha=0.5)
plt.title('NDVI vs. Income Elasticity', fontsize=20)
plt.xlabel('NDVI', fontsize=20)
plt.ylabel('Income Elasticity', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)  # Adjust major ticks
plt.tick_params(axis='both', which='minor', labelsize=20)  # Adjust minor ticks (optional)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('pics/income_elasticity_histgram_by_NDVI.png', bbox_inches='tight')

# Create two groups based on the condition
group_1 = summary_stats[summary_stats['NDVI'] >= 0.39]
group_2 = summary_stats[summary_stats['NDVI'] < 0.39]

# Set up the histogram plot
plt.figure(figsize=(8, 6))

plt.hist(group_1['median_income_elasticity'], bins=30, alpha=0.4, label='NDVI > 0.39', color='blue', edgecolor='black')

plt.hist(group_2['median_income_elasticity'], bins=30, alpha=0.4, label='NDVI < 0.39', color='red', edgecolor='black')

# Add labels and title
plt.xlabel('Income Elasticity', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.title('Income Elasticity by NDVI', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)  # Adjust major ticks
plt.tick_params(axis='both', which='minor', labelsize=20)  # Adjust minor ticks (optional)
plt.legend()
plt.savefig('pics/income_elasticity_histgram_by_NDVI.png', bbox_inches='tight')


