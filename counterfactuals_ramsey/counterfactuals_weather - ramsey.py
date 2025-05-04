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
import cobyqa
#from cobyqa import minimize
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from scipy.optimize import BFGS


demand_2018_using = pd.read_csv('demand_2018_using.csv')

p_l0 = jnp.array([3.09,  5.01,  8.54, 12.9 , 14.41])
q_l0 = jnp.array([2, 6, 11, 20])
fc_l0 = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.25+29.75])
p_l0_CAP =  jnp.array([2.37+0.05, 4.05+0.05, 6.67+0.05, 11.51+0.05, 14.21+0.05])

result = np.genfromtxt('result.csv', delimiter=',', skip_header=0)

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

A_current_outdoor = jnp.column_stack(( 
    jnp.array(demand_2018_using['bathroom']), 
                                      jnp.array( demand_2018_using['prev_NDVI']),
                                      ))
A_current_indoor = jnp.column_stack((jnp.array(demand_2018_using['bathroom']),
                                     jnp.array(demand_2018_using['above_one_acre'])
                                       ))
A_current_price = jnp.column_stack((
    jnp.array(demand_2018_using['bedroom']), 
    jnp.array(demand_2018_using['prev_NDVI']), 
    jnp.array(demand_2018_using['mean_TMAX_1']),
    jnp.array(demand_2018_using['total_PRCP'])
    ))

A_current_income = jnp.column_stack((
    jnp.array(demand_2018_using['heavy_water_app']), 
    jnp.array(demand_2018_using['bedroom']), 
    jnp.array(demand_2018_using['prev_NDVI']), 
    ))

Z_current_outdoor = jnp.column_stack((jnp.array(demand_2018_using['mean_TMAX_1']),
                                      jnp.array(demand_2018_using['IQR_TMAX_1']),
                                      jnp.array(demand_2018_using['total_PRCP']) 
                                      ,jnp.array(demand_2018_using['IQR_PRCP'])
                                      ))
Z_current_indoor = jnp.array(demand_2018_using['mean_TMAX_1'])
Z_current_indoor = Z_current_indoor[:, jnp.newaxis]
G = jnp.array(demand_2018_using['prev_NDVI'])
I = jnp.array(demand_2018_using['income'])
p0 = jnp.array(demand_2018_using['previous_essential_usage_mp'])
w_i = jnp.array(demand_2018_using['quantity'])
de = jnp.array(demand_2018_using['deflator'])
q_statusquo = jnp.array(demand_2018_using['quantity'])

q_statusquo_sum = jnp.sum(q_statusquo)

alpha = jnp.exp(jnp.dot(A_current_price, b4)
                    + c_alpha)

rho = jnp.exp(jnp.dot(A_current_income, b6)
                    + c_rho)

demand_2018_using.loc[:, 'e_alpha'] = alpha
demand_2018_using.loc[:, 'e_rho'] = rho

CAP = jnp.array(demand_2018_using['CAP_HH'])

def calculate_log_w(p_l, q_l, fc_l, Z):
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
                  #A_p = A_current_price,
                  A_o = A_current_outdoor,
                  G = G,
                  p = p_l, I = I,
                  p0 =p0, 
                  de = de,
                  ):
        A_p= jnp.column_stack((
            jnp.array(demand_2018_using['bedroom']), 
            jnp.array(demand_2018_using['prev_NDVI']), 
            Z[:, 0],
            Z[:, 2],
            ))
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
                       jnp.multiply(rho, jnp.log(jnp.maximum(I+ jnp.multiply(d_k, de), 1e-16))) + c_wo)
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


def get_eta (p_l, q_l, fc_l, d_id, quantity):
    log_w_k_base_0 = calculate_log_w_jitted(p_l, q_l, fc_l,Z_current_outdoor)
    bins = jnp.concatenate((jnp.array([0]), q_l, jnp.array([jnp.inf])))
    labels = [1, 2, 3, 4, 5]
    demand_2018_small = d_id.copy()
    demand_2018_small.loc[:, 'quantity'] = quantity
    demand_2018_small.loc[:, 'tier'] = pd.cut(demand_2018_small['quantity'], bins = bins, labels = labels)
    columns = jnp.array(demand_2018_small['tier']) - 1
    log_w_base_0 = log_w_k_base_0 [jnp.arange(log_w_k_base_0 .shape[0]), columns]
    sim = 1000
    nu = np.random.normal(loc=0.0, scale=sigma_nu, size=(log_w_base_0.size, sim))
    eta = np.random.normal(loc=0.0, scale=sigma_eta, size=(log_w_base_0.size, sim))
    log_w_base_0_sim = log_w_base_0[:, jnp.newaxis] + nu + eta
    diff_sim = jnp.log(jnp.array(demand_2018_small['quantity'])[:, jnp.newaxis]) - log_w_base_0_sim - nu
    diff_hat = jnp.mean(diff_sim, axis=1)
    diff_sd_hat = jnp.std(diff_sim, axis=1)
    demand_2018_small.loc[:, 'e_diff'] = diff_hat
    demand_2018_small.loc[:, 'sd_diff'] = diff_sd_hat
    demand_2018_small =  demand_2018_small.replace([jnp.inf, -jnp.inf], jnp.nan).dropna(subset=['e_diff'])
    demand_2018_small['mean_e_diff'] = demand_2018_small.groupby('prem_id')['e_diff'].transform('mean')
    demand_2018_small['sigma_e_diff'] = demand_2018_small.groupby('prem_id')['e_diff'].transform('std')
    #demand_2018_small = demand_2018_small.sort_values(by=['prem_id', 'bill_ym'])
    return demand_2018_small

get_eta_jitted = jax.jit(get_eta)


demand_2018_id = demand_2018_using[['prem_id', 'bill_ym']]
demand_2018_using_eta = get_eta(p_l0, q_l0, fc_l0, 
                                       demand_2018_id, 
                                       jnp.array(demand_2018_using['quantity']))

## e_diff is in log water. 

sigma_eta_df_sum = pd.DataFrame(demand_2018_using_eta.groupby('prem_id')[['mean_e_diff', 'sigma_e_diff']].mean())
sigma_eta_df_sum = sigma_eta_df_sum.groupby('prem_id').sum().reset_index()


demand_2018_using_new = pd.merge(demand_2018_using, sigma_eta_df_sum, on='prem_id')

demand_2018_using_new['e_diff'] = demand_2018_using_eta['e_diff']
demand_2018_using_new['sd_diff'] = demand_2018_using_eta['sd_diff']

demand_2018_using_eta = demand_2018_using_new[['prem_id', 'bill_ym', 'quantity','e_diff', 'sd_diff',
                                               'mean_e_diff', 'sigma_e_diff']]

demand_2018_using_eta.to_csv('demand_2018_using_eta.csv', index=False)

demand_2018_using_new.to_csv('demand_2018_using_new.csv', index=False)


#### Finish calculating HH-level eta

demand_2018_using_eta = pd.read_csv('demand_2018_using_eta.csv')

demand_2018_using_new = pd.read_csv('demand_2018_using_new.csv')


#plt.figure(figsize=(10, 6))
#demand_2018_using_eta['sigma_e_eta'].hist(bins=100, edgecolor='black')
#plt.title('Eta Distribution')
#plt.xlabel('Eta')
#plt.ylabel('Frequency')
#plt.show()



#def find_target_index(p):
 #   p_l0 = jnp.array([2.89+0.2, 4.81+0.2, 8.34+0.2, 12.70+0.2, 14.21+0.2]) 
  #  target_index = find_first_nonnegative_jit(p_l0 - p)-1
   # target_index = jnp.where(target_index<0, 0, target_index)
    #return target_index
#find_target_index_jitted = jax.jit(find_target_index)

# Create a DataFrame with the random normal variables
#demand_2018_using_eta_unique = demand_2018_using_eta.drop(columns=['bill_ym', 'quantity']).drop_duplicates().reset_index()
#demand_2018_using_eta_unique = demand_2018_using_eta.drop(columns=['bill_ym']).drop_duplicates()
np.random.seed(42)
sim = 100
#key = random.PRNGKey(42)

#random_data = np.random.normal(loc=demand_2018_using_eta_unique['mean_e_diff'].values[:, None]
 #                                  , scale=demand_2018_using_eta_unique['sigma_e_diff'].values[:, None],
  #                                 size=(demand_2018_using_eta_unique.shape[0], sim))

#random_data = np.random.normal(loc=demand_2018_using_eta['e_diff'].values[:, None]
 #                                  , scale=0.01,
                                   #, scale=demand_2018_using_eta['sigma_e_diff'].values[:, None],
  #                                 size=(demand_2018_using_eta.shape[0], sim))

#random_data = np.random.normal(loc=demand_2018_using_eta_unique['mean_e_eta'].values[:, None]
 #                                  , scale=demand_2018_using_eta_unique['sigma_e_eta'].values[:, None], size=(demand_2018_using_eta_unique.shape[0], sim))
#random_df = pd.DataFrame(random_data, columns=[f'Eta_{i+1}' for i in range(sim)]).reset_index()

#sigma_eta_df_sum = pd.concat([demand_2018_using_eta_unique, random_df], axis=1)
#sigma_eta_df_new = pd.merge(demand_2018_using_eta, sigma_eta_df_sum, on='prem_id', how='left') 
#sigma_eta_df_new = pd.concat([demand_2018_using_eta, random_df], axis=1)
len_transactions = len( demand_2018_using_eta)  
shape = (len_transactions, sim)
#nu_array = sigma_nu * random.normal(key, shape)
nu_array = np.random.normal(loc = 0, scale = sigma_nu, size = shape)
#nu_array = jnp.minimum(nu_array, 7)
#nu_array = jnp.maximum(nu_array, -7)
#eta_l = jnp.array(sigma_eta_df_new .iloc[:, -sim:])
eta_l = jnp.array(demand_2018_using_eta['e_diff'])
#eta_l = jnp.minimum(eta_l, 7)
#eta_l = jnp.maximum(eta_l, -7)


#del demand_2018_using_eta_unique, random_data, random_df, sigma_eta_df_new#, sigma_eta_df_sum

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

def cf_w (p_l, q_l, fc_l, Z,
          #nu_array = nu_array,
          eta_l = eta_l):
    log_q_sim = get_log_q_sim_jitted(p_l, q_l, fc_l, Z)
    log_q_sim = log_q_sim.reshape(len_transactions, sim)
    gc.collect()
    #nu_array =  gen_nu_array_jitted(sigma_nu)
    return log_q_sim
# + nu_array
cf_w_jitted = jax.jit(cf_w)


def get_log_w(p_l, q_l, fc_l,Z,
          nu_array = nu_array,
          eta_l = eta_l):
    log_w = calculate_log_w_jitted(p_l, q_l, fc_l, Z)
    #log_w = jnp.column_stack((log_w, eta_l))
    log_w = jnp.column_stack((log_w, nu_array))
    return log_w
get_log_w_jitted = jax.jit(get_log_w)

def get_log_q_sim(p_l, q_l, fc_l, Z,
          nu_array = nu_array,
          eta_l = eta_l):
    #nu_array = gen_nu_array_jitted(sigma_nu)
    log_w = get_log_w_jitted(p_l, q_l, fc_l, Z)
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



log_q0 = cf_w_jitted(p_l0, q_l0, fc_l0, Z_current_outdoor)
### Z_current_outdoor is basically Z_0

q0 = jnp.exp(log_q0)

def nansum_ignore_nan_inf(arr):
    mask = jnp.logical_and(jnp.isfinite(arr), ~jnp.isnan(arr))  # Mask out inf and NaN
    return jnp.sum(jnp.where(mask, arr, 0), axis = 0)
nansum_ignore_nan_inf_jitted = jax.jit(nansum_ignore_nan_inf)

def nanmean_ignore_nan_inf(arr):
    mask = jnp.logical_and(jnp.isfinite(arr), ~jnp.isnan(arr))  # Mask out inf and NaN
    return jnp.mean(jnp.where(mask, arr, 0), axis =1)
nanmean_ignore_nan_inf_jitted = jax.jit(nanmean_ignore_nan_inf)

q0_sum =nansum_ignore_nan_inf_jitted(q0)

q0_mean = nanmean_ignore_nan_inf_jitted(q0)

q0_df = pd.DataFrame(q0)

q0_df['prem_id'] = demand_2018_using_eta['prem_id']

q0_df['bill_ym'] = demand_2018_using_eta['bill_ym']

q0_df['q_statusquo'] = jnp.array(q_statusquo)

q0_df['q_sim_mean'] = jnp.array(q0_mean)

q0_df['q_diff'] = abs(q0_df['q_sim_mean'] - q0_df['q_statusquo'])
q0_df['q_diff_perct'] = abs(q0_df['q_sim_mean'] - q0_df['q_statusquo'] )/q0_df['q_statusquo']

#q0_df = q0_df.T.drop_duplicates().T

#q0_df['q_diff_perct_mean'] = q0_df.groupby('prem_id')['q_diff_perct'].mean().reset_index()

q0_df = q0_df[q0_df['q_diff_perct'] < 10]

q0_df.to_csv('q0_df.csv', index=False)

q0_df_id = q0_df[['prem_id', 'bill_ym']]

Z_current = Z_current_outdoor

demand_2018_using_eta['q_diff_perct'] = q0_df['q_diff_perct']
demand_2018_using_new['q_diff_perct'] = q0_df['q_diff_perct']

demand_2018_using_eta = demand_2018_using_eta[demand_2018_using_eta['q_diff_perct'] < 10]
demand_2018_using_new = demand_2018_using_new[demand_2018_using_new['q_diff_perct'] < 10]


demand_2018_using_eta.to_csv('demand_2018_using_eta.csv', index=False)

demand_2018_using_new.to_csv('demand_2018_using_new.csv', index=False)

############################################
#### Finish Setting UP############################################
############################################

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
import cobyqa
#from cobyqa import minimize
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from scipy.optimize import BFGS


demand_2018_using_eta = pd.read_csv('demand_2018_using_eta.csv')

demand_2018_using_new = pd.read_csv('demand_2018_using_new.csv')

demand_2018_using_eta = demand_2018_using_eta[demand_2018_using_eta['bill_ym'] >= 201901]
demand_2018_using_new = demand_2018_using_new[demand_2018_using_new['bill_ym'] >= 201901]

demand_2018_using_new.to_csv('demand_2018_using_new_small.csv', index=False)


p_l0 = jnp.array([3.09,  5.01,  8.54, 12.9 , 14.41])
q_l0 = jnp.array([2, 6, 11, 20])
fc_l0 = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.25+29.75])
p_l0_CAP =  jnp.array([2.37+0.05, 4.05+0.05, 6.67+0.05, 11.51+0.05, 14.21+0.05])

result = np.genfromtxt('result.csv', delimiter=',', skip_header=0)

p_plus1_l0 = jnp.append(p_l0[1:5],jnp.array([jnp.nan]) )
d_end0 = jnp.cumsum( (p_plus1_l0-0.2 - (p_l0-0.2)) [:4] *q_l0)
d_end0 =  jnp.insert(d_end0, 0, jnp.array([0.0]) )

p_plus1_l0_CAP = jnp.append(p_l0_CAP[1:5],jnp.array([jnp.nan]) )
d_end0_CAP = jnp.cumsum( (p_plus1_l0_CAP-0.2 - (p_l0_CAP-0.2)) [:4] *q_l0)
d_end0_CAP =  jnp.insert(d_end0_CAP, 0, jnp.array([0.0]) )

def calculate_dk (k):
    result = -fc_l0[k] + d_end0[k]
    return result
calculate_dk_jitted = jax.jit(calculate_dk)

def calculate_dk_CAP (k):
    result = -fc_l0[k] + d_end0_CAP[k]
    return result
calculate_dk_CAP_jitted = jax.jit(calculate_dk_CAP)
CAP = jnp.array(demand_2018_using_new['CAP_HH'])

def get_k0(q, q_l):
    conditions_k = [
        (q<q_l[0]),
        (( q >=q_l[0]) & (q < q_l[1])), 
        (( q >=q_l[1]) & (q < q_l[2])),
        (( q >=q_l[2]) & (q < q_l[3])),
        (q >= q_l[3]),
    ]  
    choices = [
        0,
        1,
        2,
        3,
        4
    ]
    result = jnp.select(conditions_k, choices)
    return result
get_k0_jitted = jax.jit(get_k0)

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

bedroom = jnp.array(demand_2018_using_new['bedroom'])
prev_NDVI = jnp.array(demand_2018_using_new['prev_NDVI'])

def calculate_log_w(p_l, q_l, fc_l, Z):
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
                 # A_p = A_current_price,
                  A_o = A_current_outdoor,
                  G = G,
                  p = p_l, I = I,
                  p0 =p0, 
                  de = de,
                  ):
        A_p= jnp.column_stack((
            bedroom, 
            prev_NDVI, 
            Z[:, 0],
            Z[:, 2],
            ))
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
                       jnp.multiply(rho, jnp.log(jnp.maximum(I+ jnp.multiply(d_k, de), 1e-16))) + c_wo)
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

def cf_w (p_l, q_l, fc_l, Z,
          #nu_array = nu_array,
          eta_l = eta_l):
    log_q_sim = get_log_q_sim_jitted(p_l, q_l, fc_l, Z)
    log_q_sim = log_q_sim.reshape(len_transactions, sim)
    gc.collect()
    #nu_array =  gen_nu_array_jitted(sigma_nu)
    return log_q_sim
# + nu_array
cf_w_jitted = jax.jit(cf_w)


def get_log_w(p_l, q_l, fc_l,Z,
          nu_array = nu_array,
          eta_l = eta_l):
    log_w = calculate_log_w_jitted(p_l, q_l, fc_l, Z)
    #log_w = jnp.column_stack((log_w, eta_l))
    log_w = jnp.column_stack((log_w, nu_array))
    return log_w
get_log_w_jitted = jax.jit(get_log_w)

def get_log_q_sim(p_l, q_l, fc_l, Z,
          nu_array = nu_array,
          eta_l = eta_l):
    #nu_array = gen_nu_array_jitted(sigma_nu)
    log_w = get_log_w_jitted(p_l, q_l, fc_l, Z)
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

k0 = get_k0_jitted(q_statusquo, q_l0)

d_k0 = jnp.where(CAP == 1, calculate_dk_CAP_jitted(k0), calculate_dk_jitted(k0))

q_statusquo_sum = jnp.sum(q_statusquo)

alpha = jnp.exp(jnp.dot(A_current_price, b4)
                    + c_alpha)

rho = jnp.exp(jnp.dot(A_current_income, b6)
                    + c_rho)

demand_2018_using_new.loc[:, 'e_alpha'] = alpha
demand_2018_using_new.loc[:, 'e_rho'] = rho

CAP = jnp.array(demand_2018_using_new['CAP_HH'])

Z_current = Z_current_outdoor

log_q0 = cf_w_jitted(p_l0, q_l0, fc_l0, Z_current)

'''
def cf_w_moment(p_l, q_l, fc_l, sigma_eta_df):
    log_q = cf_w(p_l, q_l, fc_l,sigma_eta_df, Z_current)
    q = jnp.exp(log_q)
    def expenditure_func(w, q_l=q_l, p_l=p_l, fc_l=fc_l):
        bins = jnp.concatenate((jnp.array([0]), q_l, jnp.array([jnp.inf])))
        binned_data = jnp.digitize(w, bins)
        q_plus1_l = jnp.insert(q_l, 0, 0)
        q_diff_l = q_l - q_plus1_l[0:4]
        cumu_sum = jnp.cumsum(p_l[0:4] * q_diff_l)
        result = jnp.where(binned_data==1, fc_l[0] + p_l[0]*w, 
                           fc_l[binned_data-1] + cumu_sum[binned_data-2] + p_l[binned_data-1] * (w - q_l[binned_data-2]))
        return result
    expenditure_func_jitted = jax.jit(expenditure_func)
    chunk_size = 100 
    num_columns = q.shape[1]
    result = []

    for start_col in range(0, num_columns, chunk_size):
        end_col = min(start_col + chunk_size, num_columns)
        data_chunk = q[:, start_col:end_col]
        result_chunk = expenditure_func_jitted(data_chunk)
        result.append(result_chunk)

    r = jnp.concatenate(result, axis=1)
    result = sigma_eta_df.copy()
    q_mean = jnp.mean(q, axis=1)
    result.loc[:, 'mean_e_q'] = q_mean
    q_std = jnp.std(q, axis=1)
    result.loc[:, 'var_e_q'] = jnp.square(q_std)
    r_mean = jnp.mean(r, axis = 1)
    result.loc[:, 'mean_e_r'] = r_mean
    r_std = jnp.std(r, axis = 1)
    result.loc[:, 'var_e_r'] = jnp.square(r_std)
    result_sum = pd.DataFrame(result.groupby('prem_id')[['mean_e_q', 'var_e_q', 'mean_e_r', 'var_e_r']].mean())
    result_sum.loc[:, 'sd_e_q'] = jnp.sqrt(jnp.array(result_sum['var_e_q']))
    result_sum.loc[:, 'sd_e_r'] = jnp.sqrt(jnp.array(result_sum['var_e_r']))
    return result_sum
#
#moment_0 = cf_w_moment(p_l0, q_l0, fc_l0, demand_2018_using_eta)

def cf_w_moment_mean(log_q, p_l, q_l, fc_l, eta_df = demand_2018_using_eta):
    #log_q = cf_w(p_l, q_l, fc_l,sigma_eta_df)
    q = jnp.exp(log_q)
    q_mean = jnp.mean(q, axis=1)
    def expenditure_func(w, q_l=q_l, p_l=p_l, fc_l=fc_l):
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
    result = eta_df.copy()
    #result.loc[:, 'mean_e_q'] = q_mean
    #q_std = jnp.std(q, axis=1)
    #result.loc[:, 'var_e_q'] = jnp.square(q_std)
    result.loc[:, 'mean_e_r'] = r_mean
    result_sum = pd.DataFrame(result.groupby('prem_id')[['mean_e_r']].mean())
    #result_sum.loc[:, 'sd_e_q'] = jnp.sqrt(jnp.array(result_sum['var_e_q']))
    return result_sum

cf_w_moment_mean_jitted = jax.jit(cf_w_moment_mean)

#moment_0_mean = cf_w_moment_mean(log_q0, p_l0, q_l0, fc_l0)

#r0 = jnp.array(moment_0_mean['mean_e_r'])[:,jnp.newaxis]

#del moment_0_mean
'''

#@jax.jit
#ef calculate_group_means(mean_e_r,  prem_id = jnp.array(demand_2018_using_eta['prem_id'], dtype = jnp.int64)):
 #   unique_prem_ids = jnp.unique(prem_id)
    
  #  def group_mean(group_id):
   #     mask = (prem_id == group_id)
    #    return jnp.mean(mean_e_r[mask])    
    #group_means = vmap(group_mean)(unique_prem_ids)
    #return group_means


####################################################
#### Changing to a different pricing scheme ########
####################################################

#param[0] = p0
#p1 = param[0] + param[1]
#prem_ids = np.array(demand_2018_using_eta['prem_id'], dtype = jnp.int64)


#### 0. Full Information, Can Predict Future
Z_current_using = jnp.column_stack((jnp.array(demand_2018_using_new['mean_TMAX_1']),
                                      jnp.array(demand_2018_using_new['IQR_TMAX_1']),
                                      jnp.array(demand_2018_using_new['total_PRCP']) 
                                      ,jnp.array(demand_2018_using_new['IQR_PRCP'])
                                      ))
Z_current = Z_current_using

#### 1. No Information, Do Nothing
#Z_current_indoor_using =jnp.zeros_like(Z_current_indoor)
#Z_current_outdoor_using =jnp.zeros_like(Z_current_outdoor)


#### 2. Avg Weather info from past 4 years 2014-2017

weather_history = pd.read_csv('weather/weather_history.csv')

weather_history_season = pd.read_csv('weather/weather_history_season.csv')

#demand_2018_using_new = pd.merge(demand_2018_using_new, weather_history, on='bill_ym', how='left' )

weather = pd.merge(weather_history, weather_history_season, on='bill_ym', how='left' )

weather_all_history = pd.read_csv('weather/weather_allhistory_1417.csv')

demand_2018_using_new_season = pd.merge(demand_2018_using_new, weather_history_season, on='bill_ym', how='left' )

demand_2018_using_new_season = pd.merge(demand_2018_using_new_season,weather_all_history , on='bill_ym', how='left' )

#Z_current_outdoor_using = jnp.column_stack((jnp.array(demand_2018_using_new['mean_Tmax_history']),
 #                                     jnp.array(demand_2018_using_new['IQR_Tmax_history']),
  #                                    jnp.array(demand_2018_using_new['sum_Prcp_history']) 
   #                                   ,jnp.array(demand_2018_using_new['IQR_Prcp_history'])))
   
   
#Z_current_indoor_using = jnp.array(demand_2018_using_new['mean_Tmax_history'])
#Z_current_indoor_using = Z_current_indoor_using[:, jnp.newaxis]
'''
Z_low = jnp.column_stack((jnp.array(demand_2018_using_new_season['mean_Tmax_history']),
                                      jnp.array(demand_2018_using_new_season['IQR_Tmax_history']),
                                      jnp.array(demand_2018_using_new_season['sum_Prcp_low']) 
                                      ,jnp.array(demand_2018_using_new_season['IQR_Prcp_low'])))

Z_high = jnp.column_stack((jnp.array(demand_2018_using_new_season['mean_Tmax_history']),
                                      jnp.array(demand_2018_using_new_season['IQR_Tmax_history']),
                                      jnp.array(demand_2018_using_new_season['sum_Prcp_high']) 
                                      ,jnp.array(demand_2018_using_new_season['IQR_Prcp_high'])))

Z_extreme_min = jnp.column_stack((jnp.array(demand_2018_using_new_season['mean_Tmax_history']),
                                      jnp.array(demand_2018_using_new_season['IQR_Tmax_history']),
                                      jnp.array(demand_2018_using_new_season['sum_Prcp_extreme_min']) 
                                      ,jnp.array(demand_2018_using_new_season['IQR_Prcp_extreme_min'])))

Z_extreme_max = jnp.column_stack((jnp.array(demand_2018_using_new_season['mean_Tmax_history']),
                                      jnp.array(demand_2018_using_new_season['IQR_Tmax_history']),
                                      jnp.array(demand_2018_using_new_season['sum_Prcp_extreme_max']) 
                                      ,jnp.array(demand_2018_using_new_season['IQR_Prcp_extreme_max'])))

Z_history = jnp.column_stack((jnp.array(demand_2018_using_new_season['mean_Tmax_history']),
                                      jnp.array(demand_2018_using_new_season['IQR_Tmax_history']),
                                      jnp.array(demand_2018_using_new_season['sum_Prcp_history']) 
                                      ,jnp.array(demand_2018_using_new_season['IQR_Prcp_history'])))

Z_zero = jnp.zeros_like(Z_history)
'''

#log_q0 = cf_w_jitted(p_l0, q_l0, fc_l0, Z_current)

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

#q0_sum =sum_ignore_outliers_jitted(q0)

def nanmean_ignore_nan_inf(arr):
    mask = jnp.logical_and(jnp.isfinite(arr), ~jnp.isnan(arr))  # Mask out inf and NaN
    return jnp.mean(jnp.where(mask, arr, 0), axis =1)
nanmean_ignore_nan_inf_jitted = jax.jit(nanmean_ignore_nan_inf)

q0_sum =nansum_ignore_nan_inf_jitted(q0)

q0_mean = nanmean_ignore_nan_inf_jitted(q0)

def get_r_mean(q_mean, p_l, q_l, fc_l):
    p_l_CAP = p_l-p_l0 + p_l0_CAP
    def expenditure_func(w, p_l, q_l, fc_l):
        bins = jnp.concatenate((jnp.array([0]), q_l, jnp.array([jnp.inf])))
        binned_data = jnp.digitize(w, bins)
        q_plus1_l = jnp.insert(q_l, 0, 0)
        q_diff_l = q_l - q_plus1_l[0:4]
        cumu_sum = jnp.cumsum(p_l[0:4] * q_diff_l)
        result = jnp.where(binned_data==1, fc_l[0] + p_l[0]*w, 
                           fc_l[binned_data-1] + cumu_sum[binned_data-2] + p_l[binned_data-1] * (w - q_l[binned_data-2]))
        return result
    expenditure_func_jitted = jax.jit(expenditure_func)
    r_mean = jnp.where(CAP==1,expenditure_func_jitted(q_mean, p_l_CAP, q_l, fc_l) , expenditure_func_jitted(q_mean, p_l, q_l, fc_l))
    return r_mean
get_r_mean_jitted = jax.jit(get_r_mean)

hh_size = len(np.unique(np.array(demand_2018_using_new_season['prem_id'], dtype = np.int64)))
sim = 100

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

def get_q_sum(p_l, q_l, fc_l, Z):
    log_q = cf_w_jitted(p_l, q_l, fc_l, Z)
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

def get_q_sum_hh(p_l, q_l, fc_l, Z):
    log_q = cf_w_jitted(p_l, q_l, fc_l, Z)
    
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

def get_q_sum_sim(p_l, q_l, fc_l, Z):
    log_q = cf_w_jitted(p_l, q_l, fc_l, Z)
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
    #q_sum_sim = sum_ignore_outliers_jitted(q)
    #q = jnp.exp(log_q)
    q_sum_sim = nansum_ignore_nan_inf_jitted(q)
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
        prem_id =  jnp.array(demand_2018_using_new_season['prem_id'], dtype = jnp.int64)
    
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

q_sum_hh0 = get_q_sum_hh_jitted(p_l0, q_l0, fc_l0, Z_current)
r0 = from_q_to_r_jitted(q_sum_hh0, p_l0, q_l0, fc_l0)

del A_current_indoor, demand_2018_using_new, demand_2018_using_eta, w_i

#Z_current_indoor

######################
#### Consumer Welfare #####
########################

def get_k(q_sum_hh, q_l):
    q_mean = q_sum_hh / sim
    conditions_k = [
        (q_mean<q_l[0]),
        (( q_mean >=q_l[0]) & (q_mean < q_l[1])), 
        (( q_mean >=q_l[1]) & (q_mean < q_l[2])),
        (( q_mean >=q_l[2]) & (q_mean < q_l[3])),
        (q_mean >= q_l[3]),
    ]  
    choices = [
        0,
        1,
        2,
        3,
        4
    ]
    result = jnp.select(conditions_k, choices)
    return result
get_k_jitted = jax.jit(get_k)


def get_virtual_income(q_sum_hh, p_l, q_l, fc_l):
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
    
    k = get_k_jitted(q_sum_hh, q_l)
    d_k = jnp.where(CAP == 1, calculate_dk_CAP_jitted(k), calculate_dk_jitted(k))
    virtual_income = jnp.maximum(jnp.multiply(d_k, de) + I, 1e-16)
    return virtual_income
get_virtual_income_jitted = jax.jit(get_virtual_income)

#alpha = jnp.exp(jnp.dot(A_current_price, b4)
 #                   + c_alpha)

def get_current_marginal_p(q_sum_hh, p_l, q_l, fc_l):
    p_l_CAP = p_l-p_l0 + p_l0_CAP
    k = get_k_jitted(q_sum_hh, q_l)
    p = jnp.where(CAP == 1,p_l_CAP[k], p_l[k])
    return p
get_current_marginal_p_jitted = jax.jit(get_current_marginal_p)

def get_expenditure_in_v_out(q_sum_hh, p_l, q_l, fc_l, Z):
    A_p= jnp.column_stack((
        bedroom, 
        prev_NDVI, 
        Z[:, 0],
        Z[:, 2],
    ))
    alpha = jnp.exp(jnp.dot(A_p, b4)
                + c_alpha
            )
    #rho = abs(jnp.dot(A_i, b6)
     #           + c_rho
      #          )
    p = get_current_marginal_p_jitted(q_sum_hh, p_l, q_l, fc_l)
    result = jnp.multiply(jnp.exp(jnp.dot(A_current_outdoor, b1) + jnp.dot(Z, b2)+ c_o + eta_l), 
                                         jnp.divide(jnp.power(p, 1-alpha), jnp.array(1-alpha)))
    return result
get_expenditure_in_v_out_jitted = jax.jit(get_expenditure_in_v_out)

def get_v_out(q_sum_hh, p_l, q_l, fc_l, Z):
    exp_v = get_expenditure_in_v_out_jitted(q_sum_hh, p_l,q_l, fc_l, Z)
    sim_result_Ik = get_virtual_income_jitted(q_sum_hh, p_l, q_l, fc_l)
    v_out = -1 *exp_v  + jnp.divide(jnp.power(sim_result_Ik, (1-rho)), (1-rho))
    return v_out
get_v_out_jitted = jax.jit(get_v_out)

### Z is the weather data being used
### Z is a nrow * 16 matrix with each column for a year of previous weather

############################################################################################################################################################
######################## Set up the baseline using historical weather  #######################################
############################################################################################################################################################

def average_Z(Z):
    n = Z.shape[0]
    reshaped = Z.reshape(n, 4, 4)
    avg_Z = reshaped.mean(axis=2)
    return avg_Z
    # The result is n by 4
average_Z_jitted = jax.jit(average_Z)

# Z history will be nrow by 16, first 4 column, 4 years of Tmax, 4 years of IQR, 4 years of prcp and 4 years of IQR

cols_to_extract = ['mean_Tmax_2014', 'mean_Tmax_2015', 'mean_Tmax_2016', 'mean_Tmax_2017',
             'IQR_Tmax_2014', 'IQR_Tmax_2015', 'IQR_Tmax_2016', 'IQR_Tmax_2017',
             'sum_Prcp_2014', 'sum_Prcp_2015', 'sum_Prcp_2016', 'sum_Prcp_2017',
             'IQR_Prcp_2014', 'IQR_Prcp_2015', 'IQR_Prcp_2016', 'IQR_Prcp_2017',]


Z_history = jnp.array(demand_2018_using_new_season[cols_to_extract].to_numpy())
Z_1417 = average_Z_jitted(Z_history)

log_qhistory = cf_w_jitted(p_l0, q_l0, fc_l0, Z_1417)
qhistory = jnp.exp(log_qhistory)

qhistory_sum =nansum_ignore_nan_inf_jitted(qhistory)
qhistory_mean = nanmean_ignore_nan_inf_jitted(qhistory)

q_sum_hhhistory = get_q_sum_hh_jitted(p_l0, q_l0, fc_l0, Z_1417)
rhistory = from_q_to_r_jitted(q_sum_hhhistory, p_l0, q_l0, fc_l0)

bill_ym = jnp.array(demand_2018_using_new_season['bill_ym'])
#unique_months = jnp.unique(bill_ym)

indices_by_month = {month: jnp.where(bill_ym % 100 == month)[0] for month in range(1, 13)}

max_len = max(len(indices) for indices in indices_by_month.values())
padded_month_indices = jnp.zeros((12, max_len), dtype=jnp.int32)
valid_month_lengths = jnp.zeros(12, dtype=jnp.int32)

for month, indices in indices_by_month.items():
    indices_int32 = jnp.array(indices, dtype=jnp.int32)  # Cast to int32
    padded_month_indices = padded_month_indices.at[month - 1, :len(indices)].set(indices_int32)
    valid_month_lengths = valid_month_lengths.at[month - 1].set(len(indices))
    
# Create a mask for valid entries
max_len = padded_month_indices.shape[1]  # Maximum possible entries in a month
mask = jnp.arange(max_len) < valid_month_lengths[:, None]

# Apply mask after slicing
mean_r0 = jnp.array([
    (r0[padded_month_indices[i]] * mask[i]).sum() / valid_month_lengths[i]
    for i in range(12)
])

min_mean_r0 = jnp.min(mean_r0)  # minimum revenue across all months
min_month_r0 = jnp.argmin(mean_r0)  # index of the month with the minimum revenue
total_r0 = min_mean_r0 * valid_month_lengths[min_month_r0]*12

 # Apply mask after slicing
mean_q0 = jnp.array([
     #(q0_mean[padded_month_indices[i]] * mask[i, :, None]).sum(axis=0) / valid_month_lengths[i]
     ( (q_sum_hh0/sim) [padded_month_indices[i]] * mask[i]).sum() / valid_month_lengths[i]
     for i in range(12)
])
max_mean_q0 = jnp.max(mean_q0)      
max_month_q0 = jnp.argmax(mean_q0)  
total_q0 = max_mean_q0 * valid_month_lengths[max_month_q0]*12

########################
#### CRRA function #####
########################
def crra(x, gamma):
    def case_log(xgamma):
        x, _ = xgamma
        return jnp.where(x > 0, jnp.log(x), jnp.nan)

    def case_power(xgamma):
        x, gamma = xgamma
        return jnp.where(x > 0, (x**(1 - gamma)) / (1 - gamma), jnp.nan)

    return lax.cond(gamma == 1.0, case_log, case_power, (x, gamma))

crra_jitted = jax.jit(crra)

######################
#### Status Quo Result #####
########################

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

#r0_filtered = r0[r0 < 20000]

#r_agg_0 = nansum_ignore_nan_inf_jitted(r0_filtered)/12
r_agg_history = nansum_ignore_nan_inf_jitted(rhistory )/12
#Array(24357931.65120108, dtype=float64)
r_agg_0 = nansum_ignore_nan_inf_jitted(r0 )/12
#Array(7258370.97836945, dtype=float64)

#q_sum_hh_1417 = get_q_sum_hh_jitted(p_l0, q_l0, fc_l0, Z_1417)
#q0_filtered = q_sum_hh_current[q_sum_hh_current < 150000]
#q0_filtered = q_sum_hh_history[q_sum_hh_history < 150000]
q_agg_history = nansum_ignore_nan_inf_jitted(q_sum_hhhistory/100)/12
# Array(1873818.19426824, dtype=float64)
q_agg_0 = nansum_ignore_nan_inf_jitted(q_sum_hh0/100)/12
#Array(794810.84743965, dtype=float64)

cs_history = get_v_out_jitted(q_sum_hhhistory , p_l0, q_l0, fc_l0, Z_1417)
#cs0_filtered = cs_0[(cs_0 > -0.5*1e9) ]
cs_agg_history = nansum_ignore_nan_inf_jitted(cs_history)/12
#Array(-16472436.102381, dtype=float64)
cs_0 = get_v_out_jitted(q_sum_hh0, p_l0, q_l0, fc_l0, Z_current)
cs_agg_0= nansum_ignore_nan_inf_jitted(cs_0)/12
#Array(-7896280.43952414, dtype=float64)

# Combine the arrays into a pandas DataFrame
detail_0 = pd.DataFrame({
    'r0': r0,
    'q_sum_hh0': q_sum_hh0/100,
    'cs_0': cs_0
})

# Export the DataFrame to a CSV file
detail_0.to_csv('ramsey_welfare_result/cs_detail_results/detail_0.csv', index=False)

# Combine the arrays into a pandas DataFrame
detail_history = pd.DataFrame({
    'rhistory': rhistory,
    'q_sum_hhhistory': q_sum_hhhistory/100,
    'cs_history': cs_history
})

# Export the DataFrame to a CSV file
detail_history.to_csv('ramsey_welfare_result/cs_detail_results/detail_history.csv', index=False)


########################
#### Revenue Conditions #####
########################
rhistory_sum_filtered =  nansum_ignore_nan_inf_jitted(rhistory)
r0_sum_filtered =  nansum_ignore_nan_inf_jitted(r0 )

rhistory_soft_constraint = 0.8*rhistory_sum_filtered

r0_soft_constraint = 0.8*r0_sum_filtered

log_r0_mean_filtered = jnp.log(r0_sum_filtered/12)

gamma = 1

crra_r0_mean_filtered = crra_jitted(r0_sum_filtered/12, gamma)
    
def revenue_compare(r, r0_benchmark =r0_sum_filtered  ):
    ### Here r is for the entire year, compared to r0 of entire year
    return r - r0_benchmark

revenue_compare_jitted = jax.jit(revenue_compare)

def revenue_compare_crra(r, r0_benchmark =crra_r0_mean_filtered  ):
    ### Here r is for the entire month, compared to r0 of entire month
    return r - r0_benchmark

revenue_compare_crra_jitted = jax.jit(revenue_compare_crra)

########################
#### Conservation Conditions #####
########################

qhistory_sum_mean = jnp.mean(qhistory_sum)
q0_sum_mean = jnp.mean(q0_sum)

q0_month_mean = q0_sum_mean/12

#q0_sum_soft_constraint = 1.1*q0_sum_mean

#q0_sum_max = jnp.mean(total_q0)

def cf_w_ci(q_sum, p_l, q_l, fc_l, q0_benchmark =q0_sum_mean):
    #log_q = cf_w(p_l, q_l, fc_l,sigma_eta_df)
    #q = jnp.exp(log_q)
    #q0 = jnp.exp(log_q0)
    #q_sum = nansum_ignore_nan_inf_jitted(q)
    #q0_sums = jnp.nansum(q0, axis = 0)
    condition= (q0_benchmark - q_sum)>0
    #condition= (0.75*q0_sum - q_sum) >0
    #count_satisfying_condition_p = jnp.count_nonzero(condition)/len(q_sum)
    return jnp.count_nonzero(condition)
cf_w_ci_jitted = jax.jit(cf_w_ci)

def conservation_condition(q_sum_sim, p_l, q_l, fc_l):
    num_satisfying = cf_w_ci_jitted(q_sum_sim, p_l, q_l, fc_l)
    #total = q_sum_sim.shape[0] * q_sum_sim.shape[1]  # 12 * 100 = 1200
    #return num_satisfying / total
    return num_satisfying/(sim)
conservation_condition_jitted = jax.jit(conservation_condition)


######################
#### Optimization #####
########################

def get_result (p_l, q_l, fc_l, Z):
    q_sum_hh = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z)
   # r = from_q_to_r_jitted(q_sum_hh, p_l, q_l, fc_l)
    cs = get_v_out(q_sum_hh, p_l, q_l, fc_l, Z)
    result =cs
    #result =cs * ( cs > -0.5 * 1e9)
    #result = jnp.zeros_like(cs)
    #result = result.at[jnp.where(mask)].set(cs[mask])
    #+ lam * r
    return result

get_result_jitted = jax.jit(get_result)

#param_no = jnp.array([3.31, 3.93, 3.36, 2.29, 2.74, 
 #                   2.52, 
  #                  3.8, 5, 7.01,
   #                 7.45, 
    #                0.01, 0.01,  18.7
     #               , 0.01
                    #1, 1
      #              ])

param0 = jnp.array([3.09, 5.01-3.09, 8.54-5.01, 12.9-8.54, 14.41-12.9, 
                   # 2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    10.8-8.5, 16.5-10.8, 37-16.5
                    , 37-37
                    ])

'''
def param_to_pq (param):
    p_l = jnp.cumsum(jnp.array([param[0], param[1], param[2], param[3], param[4]]))
    q_l = jnp.cumsum(jnp.array([ 2, param[5], param[6], param[7]]))
    q_l = jnp.minimum(q_l, q_l0)
    #q_l = jnp.cumsum(jnp.array([ param[5], param[6], param[7], param[8]]))
    #fc_l = jnp.cumsum(jnp.array([param[8], param[9], param[10], param[11], param[12]]))
    fc_l = jnp.cumsum(jnp.array([param[8], param[9], param[10], param[11], param[12]]))
    #fc_l = jnp.cumsum(jnp.array([8.5, param[8], param[9], param[10], param[11]]))
    fc_l = jnp.minimum(fc_l, 37*2)
    #lam1 = param[8]
    #fc_l = fc_l0
    #### Wilson's condition
    #fc_l = jnp.cumsum(jnp.array([fc_l0[0], (p_l[1] - p_l[0])*q_l[0], 
     #                           (p_l[2] - p_l[1])*(q_l[1]-q_l[0]), 
      #                          (p_l[3] - p_l[2])*(q_l[2]-q_l[1]), 
       #                         (p_l[4] - p_l[3])*(q_l[3]-q_l[2])]))
    #lam2 = param[9]
    return p_l, q_l, fc_l
    #return p_l, q_l, lam1, lam2
param_to_pq_jitted = jax.jit(param_to_pq)
'''
def param_to_pq0 (param):
    p_l = jnp.cumsum(jnp.array([param[0], param[1], param[2], param[3], param[4]]))
    #q_l = jnp.cumsum(jnp.array([ 2, param[5], param[6], param[7]]))
    q_l = jnp.cumsum(jnp.array([ param[5], param[6], param[7], param[8]]))
    #q_l = jnp.minimum(q_l, q_l0)
    #fc_l = jnp.cumsum(jnp.array([param[8], param[9], param[10], param[11], param[12]]))
    fc_l = jnp.cumsum(jnp.array([param[9], param[10], param[11], param[12], param[13]]))
    #fc_l = jnp.cumsum(jnp.array([8.5, param[8], param[9], param[10], param[11]]))
    #fc_l = jnp.minimum(fc_l, 37*2)
    #lam1 = param[8]
    #fc_l = fc_l0
    #### Wilson's condition
    #fc_l = jnp.cumsum(jnp.array([fc_l0[0], (p_l[1] - p_l[0])*q_l[0], 
     #                           (p_l[2] - p_l[1])*(q_l[1]-q_l[0]), 
      #                          (p_l[3] - p_l[2])*(q_l[2]-q_l[1]), 
       #                         (p_l[4] - p_l[3])*(q_l[3]-q_l[2])]))
    #lam2 = param[9]
    return p_l, q_l, fc_l
    #return p_l, q_l, lam1, lam2
param_to_pq0_jitted = jax.jit(param_to_pq0)

#p_l0, q_l0, fc_l0 = param_to_pq_jitted(param0)
'''
def objective(param, lam):
    param = jnp.maximum(param, 0.01)
    p_l, q_l, fc_l = param_to_pq_jitted(param)
    jax.debug.print("Current param {y}", y= jax.device_get(param))
    result_low = get_result_jitted(p_l, q_l, fc_l, Z_low, lam)
    result_high = get_result_jitted(p_l, q_l, fc_l, Z_high, lam)
    result = (result_low + result_high)/2
    #result = get_result_jitted(p_l, q_l, fc_l, Z_history, lam)
    result = -1 * nansum_ignore_nan_inf_jitted(result)
    #result = -1 * sum_ignore_outliers_jitted(result)
    result_value = jax.device_get(result)
    jax.debug.print("Current Value {x}", x= result_value)
    return result
objective_jitted = jax.jit(objective)
'''

def objective0(param, Z):
    param = jnp.maximum(param, 0.01)
    p_l, q_l, fc_l = param_to_pq0_jitted(param)
    jax.debug.print("Current param {y}", y= jax.device_get(param))
    processed_Z = average_Z_jitted(Z)
    del Z
    #result_low = get_result_jitted(p_l, q_l, fc_l, Z_low, lam)
    #result_high = get_result_jitted(p_l, q_l, fc_l, Z_high, lam)
    #result = (result_low + result_high)/2
    result = get_result_jitted(p_l, q_l, fc_l, processed_Z)
    result = -1 * nansum_ignore_nan_inf_jitted(result)
    #result = -1 * sum_ignore_outliers_jitted(result)
    result_value = jax.device_get(result)
    jax.debug.print("Current Value {x}", x= result_value)
    return result
objective0_jitted = jax.jit(objective0)
'''



def revenue_non_exceeding_constraint(param):
    p_l, q_l, fc_l = param_to_pq_jitted(param)
    q_sum_hh = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_low)
    r = from_q_to_r_jitted(q_sum_hh, p_l, q_l, fc_l)
    return revenue_compare_jitted(r)

revenue_non_exceeding_constraint_jitted = jax.jit(revenue_non_exceeding_constraint)
'''
#r_benchmark = r0_soft_constraint
def revenue_lower_bound_constraint(param, Z):
    p_l, q_l, fc_l = param_to_pq0_jitted(param)
    ## Take Average
    processed_Z = average_Z_jitted(Z)
    del Z
    #max_prcp = jnp.max(processed_Z[:,2])
    #max_month = jnp.where(processed_Z[:,2] == max_prcp)[0]
    q_sum_hh = get_q_sum_hh_jitted(p_l, q_l, fc_l, processed_Z)
    r = from_q_to_r_jitted(q_sum_hh, p_l, q_l, fc_l)
    del q_sum_hh


    r_sum_filtered = nansum_ignore_nan_inf_jitted(r)
    result = revenue_compare_jitted(r_sum_filtered)
    """
    # Create a mask for valid entries
    max_len = padded_month_indices.shape[1]  # Maximum possible entries in a month
    mask = jnp.arange(max_len) < valid_month_lengths[:, None]

    # Apply mask after slicing
    mean_r = jnp.array([
        (r[padded_month_indices[i]] * mask[i]).sum() / valid_month_lengths[i]
        for i in range(12)
    ])

    lowest_months = jnp.argsort(mean_r)[:1]    
    lowest_mean_r = mean_r[lowest_months] 
    #min_mean_r = jnp.min(mean_r)  # minimum revenue across all months
    #min_month = jnp.argmin(mean_r)  # index of the month with the minimum revenue
    #total_r = min_mean_r * valid_month_lengths[min_month]*12gc.col

    # Compute the revenue difference from the benchmark
    diff_from_benchmark = mean_r - r0_sum_filtered
    # Get the indices of months where mean_r is below the benchmark
    valid_mask = diff_from_benchmark < 0
    valid_indices = jnp.where(valid_mask, size=12)[0]  # Get valid indices
    mean_r_valid = jnp.take(mean_r, valid_indices)  # Select valid revenue values
    valid_lengths = jnp.take(valid_month_lengths, valid_indices)  # Select valid month lengths

    total_r = jnp.where(
        jnp.any(valid_mask),
        jnp.mean(mean_r_valid * valid_lengths) * 12,
        lowest_mean_r * valid_month_lengths[lowest_months] * 12
    )
    """
    """    
    # Get indices of the 4 lowest mean revenues
    lowest_months = jnp.argsort(mean_r)[:4]  
    # Get the 4 lowest mean revenue values
    lowest_mean_r = mean_r[lowest_months]
    # Compute total revenue using these months
    total_r = jnp.mean(lowest_mean_r * valid_month_lengths[lowest_months])*12 
    result = revenue_compare_jitted(total_r)
    """
    #q_sum_hh_low = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_low)
    #r_low = from_q_to_r_jitted(q_sum_hh_low, p_l, q_l, fc_l)
    #del q_sum_hh_low
    #r = (r_high + r_low)/2
    #r = r_high
    
    return result

revenue_lower_bound_constraint_jitted = jax.jit(revenue_lower_bound_constraint)


def revenue_lower_bound_crra_constraint(param, Z):
    p_l, q_l, fc_l = param_to_pq0_jitted(param)
    ## Take Average
    processed_Z = average_Z_jitted(Z)
    del Z
    #max_prcp = jnp.max(processed_Z[:,2])
    #max_month = jnp.where(processed_Z[:,2] == max_prcp)[0]
    q_sum_hh = get_q_sum_hh_jitted(p_l, q_l, fc_l, processed_Z)
    r = from_q_to_r_jitted(q_sum_hh, p_l, q_l, fc_l)
    del q_sum_hh
    # Create a mask for valid entries
    max_len = padded_month_indices.shape[1]  # Maximum possible entries in a month
    mask = jnp.arange(max_len) < valid_month_lengths[:, None]

    # Apply mask after slicing
    total_r = jnp.array([
        (r[padded_month_indices[i]] * mask[i]).sum()
        for i in range(12)
    ]) 
    crra_total_r = crra_jitted(jnp.maximum(total_r, 1e-16), gamma)   
    result = revenue_compare_crra_jitted(jnp.sum(crra_total_r)/12) 

    
    return result

revenue_lower_bound_crra_constraint_jitted = jax.jit(revenue_lower_bound_crra_constraint)



def get_q_sum_sim_max(p_l, q_l, fc_l, Z):
    log_q = cf_w_jitted(p_l, q_l, fc_l, Z)
    q = jnp.exp(log_q)
    del log_q
    max_len = padded_month_indices.shape[1]  # Maximum possible entries in a month
    mask = jnp.arange(max_len) < valid_month_lengths[:, None]
    
    # Compute sum_q while applying the mask
    sum_q = jnp.array([
        (q[jnp.take(padded_month_indices, i, axis=0)] * mask[i, :, None]).sum(axis=0)
        for i in range(12)
    ])

    # Sort to get indices of the top 4 months
    top_4_indices = jnp.argpartition(sum_q, -4, axis=0)[-4:]  # Efficient selection
    top_4_q = jnp.take_along_axis(sum_q, top_4_indices, axis=0)

    total_q = jnp.mean(top_4_q, axis=0) * 12
    
    return total_q
get_q_sum_sim_max_jitted = jax.jit(get_q_sum_sim_max)

def get_q_sum_sim_month(p_l, q_l, fc_l, Z):
    log_q = cf_w_jitted(p_l, q_l, fc_l, Z)
    q = jnp.exp(log_q)
    del log_q
    max_len = padded_month_indices.shape[1]  # Maximum possible entries in a month
    mask = jnp.arange(max_len) < valid_month_lengths[:, None]
    
    # Compute sum_q while applying the mask
    sum_q = jnp.array([
        (q[jnp.take(padded_month_indices, i, axis=0)] * mask[i, :, None]).sum(axis=0)
        for i in range(12)
    ])
    
    return sum_q

get_q_sum_sim_month_jitted = jax.jit(get_q_sum_sim_month)

def conservation_constraint(param, Z):
    #param = jnp.maximum(param, 0)
    p_l, q_l, fc_l = param_to_pq0_jitted(param)
    #jax.debug.print("Current param {y}", y= jax.device_get(param))
    #result = nansum_ignore_nan_inf_jitted(q_sum_sim - q0_sum)/len(q0_sum)
    #q_sum_sim_high = get_q_sum_sim_jitted(p_l, q_l, fc_l, Z_high)
    #result_high = conservation_condition_jitted(q_sum_sim_high, p_l, q_l, fc_l)
    #del q_sum_sim_high
    processed_Z = average_Z_jitted(Z)
    del Z

    q_sum_sim = get_q_sum_sim_jitted(p_l, q_l, fc_l, processed_Z)
    result = conservation_condition_jitted(q_sum_sim, p_l, q_l, fc_l)
    del q_sum_sim
  
    """ 
    q_sum_sim_month = get_q_sum_sim_month_jitted(p_l, q_l, fc_l, processed_Z) # 12*sim dimension array
    result = conservation_condition_jitted(q_sum_sim_month, p_l, q_l, fc_l)
    del q_sum_sim_month

    """ 
    # Create a mask for valid entries
    #max_len = padded_month_indices.shape[1]  # Maximum possible entries in a month
    #mask = jnp.arange(max_len) < valid_month_lengths[:, None]
    # Apply mask after slicing
    #mean_q = jnp.array([
    #    (q[padded_month_indices[i]] * mask[i]).sum() / valid_month_lengths[i]
    #    for i in range(12)
    #])
    
    #max_mean_q = jnp.max(mean_q)  # minimum revenue across all months
    #max_month = jnp.argmax(mean_q)  # index of the month with the minimum revenue
    #total_q = max_mean_q * valid_month_lengths[max_month]*12
    #result = total_q0 - total_q
    """ 
    q_max_sum_sim = get_q_sum_sim_max_jitted(p_l, q_l, fc_l, processed_Z)
    result = conservation_condition_jitted(q_max_sum_sim, p_l, q_l, fc_l)
    del q_max_sum_sim
    """
    #result = (result_high + result_low)/2
    #result = result_low
    #result_value = jax.device_get(result)
    #jax.debug.print("Current Value {x}", x= result_value)
    return result

conservation_constraint_jitted = jax.jit(conservation_constraint)


#constraint2 = NonlinearConstraint(revenue_non_exceeding_constraint_jitted, 
 #                                0.0, jnp.inf, jac='2-point', hess=BFGS())
 
#constraint1 = NonlinearConstraint( conservation_constraint_jitted, 
 #                                0.0,jnp.inf , jac='2-point', hess=BFGS())

constraint1 = NonlinearConstraint(lambda x: conservation_constraint_jitted(x, 1), 
                                 0.95, 1.0, jac='2-point', hess=BFGS())

constraint2 = NonlinearConstraint(revenue_lower_bound_constraint_jitted, 
                                 0.0, jnp.inf, jac='2-point', hess=BFGS())

bounds0 = Bounds([0.01, 0.01, 0.01, 0.01, 0.01, 
                 0.01, 
                 0.01, 0.01, 0.01,
                 0.01, 
                0.01, 0.01, 0.01,
                0.01
                ], 
                [#20, 20, 20, 20, 20, 
                 jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf,
                 jnp.inf, 
                 #4, 9, 14,
                 jnp.inf, jnp.inf, jnp.inf, 
                 jnp.inf, 
                 jnp.inf, jnp.inf, jnp.inf, 
                 jnp.inf
                # 20, 
                 #20, 20, 20, 
                 #20
                 ])

'''
bounds = Bounds([0.01, 0.01, 0.01, 0.01, 0.01, 
                 #0.01, 
                 0.01, 0.01, 0.01,
                 0.01, 
                0.01, 0.01, 0.01,
                0.01
                ], 
                [20, 20, 20, 20, 20, 
                 #jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf,
                 #jnp.inf, 
                 4, 9, 14,
                 #jnp.inf, jnp.inf, jnp.inf, 
                 #jnp.inf, 
                 #jnp.inf, jnp.inf, jnp.inf, 
                 #jnp.inf
                 20, 
                 20, 20, 20, 
                 20
                 ])
'''
'''
param0 = jnp.array([3.09, 5.01-3.09, 8.54-5.01, 12.9-8.54, 14.41-12.9, 
                   # 2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    10.8-8.5, 16.5-10.8, 37-16.5
                    , 37-37
                    ])
'''

param0_2 = jnp.array([3.09, 5.01-3.09, 8.54-5.01, 12.9-8.54, 14.41-12.9, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    10.8-8.5, 16.5-10.8, 37-16.5
                    , 37-37
                    ])
"""
#For extreme bound 
param0_high = jnp.array([3.09, 50, 50, 50, 50, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    50, 50, 50
                    , 50
                    ])

#For soft extreme bound 
param0_high = jnp.array([3, 15, 15, 15, 15, 
                    2, 
                    6-2, 11-6, 20-11,
                    8, 
                    50, 50, 50
                    , 50
                    ])

# For average bound 
param0_high = jnp.array([3, 8, 8, 8, 8, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    10, 10, 10
                    , 10
                    ])

# For welfareplus bound 
param0_high = jnp.array([3, 6, 6, 6, 6, 
                    2, 
                    6-2, 11-6, 20-11,
                    8, 
                    10, 10, 10
                    , 10
                    ])

# For CRRA log revenue bound
param0_high = jnp.array([3, 15, 15, 15, 15, 
                    2, 
                    6-2, 11-6, 20-11,
                    8, 
                    20, 20, 20
                    , 20
                    ])

# For crra gamma = 0.5 bound
param0_high = jnp.array([3, 10, 10, 10, 10, 
                    2, 
                    6-2, 11-6, 20-11,
                    8, 
                    12, 12, 12
                    , 12
                    ])

"""
param0_high = jnp.array([3, 10, 10, 10, 10, 
                    2, 
                    6-2, 11-6, 20-11,
                    8, 
                    12, 12, 12
                    , 12
                    ])
#p_l, q_l, fc_l = param_to_pq0_jitted(param0)



'''
q_sum_hh_low = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_low)
r_low_0 = from_q_to_r_jitted(q_sum_hh_low, p_l, q_l, fc_l)
r_low_agg_0 = sum_ignore_outliers_jitted(r_low_0)/12
cs_low_0 = get_v_out_jitted(q_sum_hh_low, p_l, q_l, fc_l)
cs_low_agg_0 = sum_ignore_outliers_jitted(cs_low_0)/12

q_sum_hh_high = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_high)
r_high_0 = from_q_to_r_jitted(q_sum_hh_high, p_l, q_l, fc_l)
r_high_agg_0 = sum_ignore_outliers_jitted(r_high_0)/12
cs_high_0 = get_v_out_jitted(q_sum_hh_high, p_l, q_l, fc_l)
cs_high_agg_0 = sum_ignore_outliers_jitted(cs_high_0)/12

q_sum_hh_history = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_history)
r_history_0 = from_q_to_r_jitted(q_sum_hh_history, p_l, q_l, fc_l)
r_history_agg_0 = sum_ignore_outliers_jitted(r_history_0)/12
cs_history_0 = get_v_out_jitted(q_sum_hh_history, p_l, q_l, fc_l)
cs_history_agg_0 = sum_ignore_outliers_jitted(cs_history_0)/12

del r_low_0, cs_low_0, r_high_0, cs_high_0

del r_history_0, cs_history_0
'''
####### set t1 = 2, t2, t3, t4 have upper bounds, CS, with revenue non exceeding constraint
'''
# First optimization attempt
solution1 = cobyqa.minimize(lambda x: objective(x, 1), 
                            param0, 
                            bounds=bounds, 
                            constraints=(constraint1,
                                         constraint2
                                         ), 
                            options={'disp': True, 
                                     'feasibility_tol': 0.01, 
                                     'radius_final': 0.01})

# Check convergence and rerun if necessary
if not solution1.success:
    print("First attempt did not converge. Retrying with new initial guess.")
    solution1_2 = cobyqa.minimize(lambda x: objective(x, 1), 
                                solution1.x, 
                                bounds=bounds, 
                                constraints=(constraint1,
                                             constraint2
                                             ), 
                                options={'disp': True, 
                                         'feasibility_tol': 0.01, 
                                         'radius_final': 0.01})
    solution1_final = solution1_2
else:
    print("Optimization converged successfully.")
    solution1_final = solution1

# Use the converged result
p_l, q_l, fc_l = param_to_pq_jitted(solution1_final.x)

q_sum_hh_low = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_low)
r_low1 = from_q_to_r_jitted(q_sum_hh_low, p_l, q_l, fc_l)
r_low1_agg = sum_ignore_outliers_jitted(r_low1)/12
cs_low1 = get_v_out_jitted(q_sum_hh_low, p_l, q_l, fc_l)
cs_low1_agg = sum_ignore_outliers_jitted(cs_low1)/12

q_sum_hh_high = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_high)
r_high1 = from_q_to_r_jitted(q_sum_hh_high, p_l, q_l, fc_l)
r_high1_agg = sum_ignore_outliers_jitted(r_high1)/12
cs_high1 = get_v_out_jitted(q_sum_hh_high, p_l, q_l, fc_l)
cs_high1_agg = sum_ignore_outliers_jitted(cs_high1)/12

del r_low1, cs_low1, r_high1, cs_high1

####  Model 0 ########  1*
#### no_weather_ramsey_history
#p_l = jnp.array([1.94,1.95,1.96,1.97,1.98])
#q_l = jnp.array([2.00,2.01,2.03,13.40])
#fc_l = jnp.array([20.00,39.91,55.07,75.07,76.66])

#p_l = jnp.array([0.01,0.02,0.03,0.04,1.99])
#q_l = jnp.array([4.29,11.25,18.06,38.27])
#fc_l = jnp.array([16.16,26.84,34.66,55.26,58.61])

####  Model 1 ########  6*
#### basic_ramsey_avg_exp
#p_l = jnp.array([0.01,0.02,0.03,2.78,2.79])
#q_l = jnp.array([2.00,6.00,14.33,22.81])
#fc_l = jnp.array([19.75,35.12,51.26,71.26,76.46])

#p_l = jnp.array([0.01,0.02,0.03,0.04,1.90])
#q_l = jnp.array([1.70,10.45,18.10,39.02])
#fc_l = jnp.array([20.55,31.89,41.31,59.49,59.51])


####  Model 1.5 ########  8*
#### basic_ramsey_worst_extreme
#p_l = jnp.array([0.01,0.02,0.03,2.74,2.75])
#q_l = jnp.array([2.00,4.86,11.05,20.03])
#fc_l = jnp.array([20.00,37.20,57.20,76.90,76.91])

#p_l = jnp.array([0.69,0.70,0.71,0.72,3.45])
#q_l = jnp.array([1.12,4.14,7.24,22.39])
#fc_l = jnp.array([16.64,31.41,42.05,63.29,64.75])


####  Model 2.0 ########  9*
#### conservation_noweather_ramsey_history
#p_l = jnp.array([3.36,5.25,7.86,11.54,13.13])
#q_l = jnp.array([2.00,6.00,10.88,19.81])
#fc_l = jnp.array([9.44,11.98,18.18,38.18,38.21])

#p_l = jnp.array([3.05,4.71,7.83,12.07,13.56])
#q_l = jnp.array([3.00,6.84,11.74,20.62])
#fc_l = jnp.array([8.44,10.80,16.60,36.96,37.15])

####  Model 2.1 ########  10*
#### conservation_baseline_ramsey_avg_exp
#p_l = jnp.array([5.49,5.50,6.37,9.89,9.90])
#q_l = jnp.array([2.00,5.95,11.64,20.34])
#fc_l = jnp.array([10.58,13.73,19.79,39.79,39.88])

#p_l = jnp.array([3.06,4.73,8.00,12.12,13.57])
#q_l = jnp.array([3.00,6.89,11.87,20.84])
#fc_l = jnp.array([8.49,10.76,16.37,36.97,36.99])


####  Model 2.15 ########  11*
#### conservation_baseline_ramsey_worst_extreme
#p_l = jnp.array([3.51,5.16,8.16,11.89,13.03])
#q_l = jnp.array([2.00,5.68,10.62,19.62])
#fc_l = jnp.array([8.93,11.89,17.89,37.89,38.35])

#p_l = jnp.array([3.09,4.94,7.62,11.54,12.92])
#q_l = jnp.array([3.01,6.76,11.67,20.75])
#fc_l = jnp.array([8.75,10.92,16.67,37.25,37.26])


'''
'''
weather_input = demand_2018_using_new_season[['mean_Tmax_history', 'IQR_Tmax_history', 'sum_Prcp_history','sum_Prcp_extreme_min','sum_Prcp_extreme_max',
                                              'IQR_Prcp_history','IQR_Prcp_extreme_min','IQR_Prcp_extreme_max']].to_numpy()

@jax.jit
def calculate_Z_step(step, mode_flag, weather_data=weather_input):
    """
    Construct Z_step based on the mode_flag (0 for 'mean', 1 for 'iqr').
    """
    def compute_mean():
        return jnp.column_stack((
            weather_data[:, 0],
            weather_data[:, 1],
            jnp.minimum(
                jnp.maximum(
                    weather_data[:, 2] + step,
                    weather_data[:, 3]
                ),
                weather_data[:, 4]
            ),
            weather_data[:, 5]
        ))

    def compute_iqr():
        return jnp.column_stack((
            weather_data[:, 0],
            weather_data[:, 1],
            weather_data[:, 2],
            jnp.minimum(
                jnp.maximum(
                    weather_data[:, 5] + step,  # Fixed here: adding `step` to weather_data[:, 5] 
                    weather_data[:, 6]
                ),
                weather_data[:, 7]
            )
        ))

    return jax.lax.cond(mode_flag == 0, compute_mean, compute_iqr)


@jax.jit
def compute_aggregates_for_mode(steps, mode_flag, p_l, q_l, fc_l):
    """
    Compute aggregates for either mean or IQR mode.
    """
    r_agg_results = jnp.zeros(len(steps))
    cs_agg_results = jnp.zeros(len(steps))
    #cs_all_steps = []  # For storing each step, without using large lists

    # Iterate over steps and compute aggregates
    for i, step in enumerate(steps):
        Z_step = calculate_Z_step(step, mode_flag)
        q_sum_hh_step = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_step)
        r_step = from_q_to_r_jitted(q_sum_hh_step, p_l, q_l, fc_l)
        cs_step = get_v_out_jitted(q_sum_hh_step, p_l, q_l, fc_l)

        # Store results directly into the pre-allocated arrays
        r_agg_results = r_agg_results.at[i].set(sum_ignore_outliers_jitted(r_step) / 12)
        cs_agg_results = cs_agg_results.at[i].set(sum_ignore_outliers_jitted(cs_step) / 12)

        # Append cs_step to the list (we could also pre-allocate if known size)
        #cs_all_steps.append(cs_step)

    return r_agg_results, cs_agg_results
'''
'''

r_step_agg_results = []
cs_step_agg_results = []
cs_step_results = []

for step in steps:
    # Update Z_step with the current step
    Z_step = jnp.column_stack((
        jnp.array(demand_2018_using_new_season['mean_Tmax_history']),
        jnp.array(demand_2018_using_new_season['IQR_Tmax_history']),
       # jnp.minimum(
            jnp.maximum(
                jnp.array(demand_2018_using_new_season['sum_Prcp_history']) + step,
                0
            ),
         #   jnp.array(demand_2018_using_new_season['sum_Prcp_extreme_max'])
        #),
        jnp.array(demand_2018_using_new_season['IQR_Prcp_history'])
    ))
    
    # Compute quantities
    q_sum_hh_step = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_step)
    r_step = from_q_to_r_jitted(q_sum_hh_step, p_l, q_l, fc_l)
    #r_step_agg = sum_ignore_outliers_jitted(r_step) / 12
    r_step_agg = nansum_ignore_nan_inf_jitted(r_step) / 12
    cs_step = get_v_out_jitted(q_sum_hh_step, p_l, q_l, fc_l)
    #cs_step_agg = sum_ignore_outliers_jitted(cs_step) / 12
    cs_step_agg = nansum_ignore_nan_inf_jitted(cs_step) / 12
    del q_sum_hh_step, r_step
    
    # Append results
    r_step_agg_results.append(r_step_agg)
    cs_step_agg_results.append(cs_step_agg)
    cs_step_results.append(cs_step)
    del cs_step
    gc.collect()

# Convert results to arrays for further processing
r_step_agg_results = jnp.array(r_step_agg_results)
cs_step_agg_results = jnp.array(cs_step_agg_results)
cs_step_results = jnp.array(cs_step_results)
cs_step_results=cs_step_results.T
cs_step_results_df = pd.DataFrame(cs_step_results)

# Save the DataFrame to a CSV file
cs_step_results_df.to_csv("ramsey_welfare_result/cs_detail_results/conservation_weather_zero_cs_mean.csv", index=False)

del cs_step_results, cs_step_results_df

r_step_agg_results_iqr = []
cs_step_agg_results_iqr = []
cs_step_results_iqr = []

for step in s_steps:
    # Update Z_step with the current step
    Z_step = jnp.column_stack((
        jnp.array(demand_2018_using_new_season['mean_Tmax_history']),
        jnp.array(demand_2018_using_new_season['IQR_Tmax_history']),
        jnp.array(demand_2018_using_new_season['sum_Prcp_history']),
        #jnp.minimum(
            jnp.maximum(
                jnp.array(demand_2018_using_new_season['IQR_Prcp_history']) + step,
                0
            ),
            #jnp.array(demand_2018_using_new_season['IQR_Prcp_extreme_max'])
        #)
    ))
    
    # Compute quantities
    q_sum_hh_step = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_step)
    r_step = from_q_to_r_jitted(q_sum_hh_step, p_l, q_l, fc_l)
    #r_step_agg = sum_ignore_outliers_jitted(r_step) / 12
    r_step_agg = nansum_ignore_nan_inf_jitted(r_step) / 12
    cs_step = get_v_out_jitted(q_sum_hh_step, p_l, q_l, fc_l)
    #cs_step_agg = sum_ignore_outliers_jitted(cs_step) / 12
    cs_step_agg = nansum_ignore_nan_inf_jitted(cs_step) / 12
    del q_sum_hh_step, r_step
    
    # Append results
    r_step_agg_results_iqr.append(r_step_agg)
    cs_step_agg_results_iqr.append(cs_step_agg)
    cs_step_results_iqr.append(cs_step)
    del cs_step
    gc.collect()

# Convert results to arrays for further processing
r_step_agg_results_iqr = jnp.array(r_step_agg_results_iqr)
cs_step_agg_results_iqr = jnp.array(cs_step_agg_results_iqr)
cs_step_results_iqr = jnp.array(cs_step_results_iqr)
cs_step_results_iqr=cs_step_results_iqr.T
cs_step_results_iqr_df = pd.DataFrame(cs_step_results_iqr)

# Save the DataFrame to a CSV file
cs_step_results_iqr_df.to_csv("ramsey_welfare_result/cs_detail_results/conservation_weather_zero_cs_iqr.csv", index=False)

del cs_step_results_iqr, cs_step_results_iqr_df

# Convert results to DataFrame
results_df = pd.DataFrame({
    "steps": steps,
    "r_step_agg_mean": r_step_agg_results,
    "cs_step_agg_mean": cs_step_agg_results,
    "r_step_agg_var": r_step_agg_results_iqr,
    "cs_step_agg_var": cs_step_agg_results_iqr,
})


results_df.to_csv("ramsey_welfare_result/conservation_weather_zero_result.csv", index=False)

'''
@jax.jit
def loop_Z_get_price(Z, param0_2 = param0_high):
    constraint_conserve = NonlinearConstraint(
        lambda x: conservation_constraint_jitted(x, Z), 
        0.95, 1.0, jac='2-point', hess=BFGS()
    )
    constraint_revenue = NonlinearConstraint(
        lambda x: revenue_lower_bound_constraint_jitted(x, Z), 
        0.0, jnp.inf, jac='2-point', hess=BFGS()
    )

    param0_2 = param0_2.astype(float)
    param0_2_np = np.array(param0_2)

    solution1_nobd = cobyqa.minimize(
        lambda x: objective0(x, Z), 
        param0_2_np,
        bounds=bounds0, 
        constraints=(constraint_conserve, constraint_revenue), 
        options={'disp': True, 'feasibility_tol': 0.01, 'radius_final': 0.001}
    )

     #Check convergence and retry if necessary
    solution1_nobd.x = np.array(solution1_nobd.x)
    if not solution1_nobd.success:
        print("First attempt did not converge. Retrying with new initial guess.")
        solution1_nobd_2 = cobyqa.minimize(
            lambda x: objective0(x, Z), 
            np.array(solution1_nobd.x, dtype=float),  # Ensure NumPy conversion
            bounds=bounds0, 
            constraints=(constraint_conserve, constraint_revenue), 
            options={'disp': True, 'feasibility_tol': 0.01, 'radius_final': 0.001}
        )
        solution1_nobd_final = solution1_nobd_2
    else:
        print("Optimization converged successfully.")
        solution1_nobd_final = solution1_nobd

    #solution1_nobd_final = solution1_nobd 
    p_l, q_l, fc_l = param_to_pq0_jitted(solution1_nobd_final.x)
    return p_l, q_l, fc_l

#loop_Z_get_price_jitted = jax.jit(loop_Z_get_price)

# Use the best result
#p_l, q_l, fc_l = param_to_pq_jitted(solution1_nobd_final.x)
'''
q_sum_hh_low = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_low)
r_low1_nobd = from_q_to_r_jitted(q_sum_hh_low, p_l, q_l, fc_l)
r_low1_nobd_agg = sum_ignore_outliers_jitted(r_low1_nobd)/12
cs_low1_nobd = get_v_out_jitted(q_sum_hh_low, p_l, q_l, fc_l)
cs_low1_nobd_agg = sum_ignore_outliers_jitted(cs_low1_nobd)/12

q_sum_hh_high = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_high)
r_high1_nobd = from_q_to_r_jitted(q_sum_hh_high, p_l, q_l, fc_l)
r_high1_nobd_agg = sum_ignore_outliers_jitted(r_high1_nobd)/12
cs_high1_nobd = get_v_out_jitted(q_sum_hh_high, p_l, q_l, fc_l)
cs_high1_nobd_agg = sum_ignore_outliers_jitted(cs_high1_nobd)/12

del r_low1_nobd, cs_low1_nobd, r_high1_nobd, cs_high1_nobd
'''
#p_l = jnp.array([3.05,4.81,7.83,11.84,13.19])
#q_l = jnp.array([3.00,6.93,11.92,20.79])
#fc_l = jnp.array([8.46,10.73,16.48,36.96,37.11])
#p_l, q_l, fc_l = param_to_pq_jitted(param0)

steps = jnp.arange(-0.25, 0.3, 0.05)

#s_steps = jnp.arange(-0.05,0.35-0.025 , 0.025)

# Lists to store results
pl_step_agg_results = []
ql_step_agg_results = []
fcl_step_agg_results = []
r_step_agg_results = []
cs_step_agg_results = []
q_step_agg_results = []
cs_step_results = []
q_step_results = []
r_step_results = []
error_log = []  # Store errors

param0_high = jnp.array([3, 8, 8, 8, 8, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    10, 10, 10
                    , 10
                    ])

Z_current_duplicate = jnp.repeat(Z_current, repeats=4, axis=1)

for step in steps:
    try:
        # Update Z_step with the current step
        Z_step = Z_current_duplicate.copy()  # Preserve original structure
        #Z_step = Z_history.copy()  # Preserve original structure
        Z_step = Z_step.at[:, 8:12].add(step)  # Modify slice (columns 8 to 11)
        Z_step = jnp.maximum(Z_step, 1e-16)

        # Define constraints
        constraint_conserve = NonlinearConstraint(
            lambda x: conservation_constraint_jitted(x, Z_step), 
            0.95, 1.0, jac='2-point', hess=BFGS()
        )
        constraint_revenue = NonlinearConstraint(
            lambda x: revenue_lower_bound_constraint_jitted(x, Z_step), 
            0.0, jnp.inf, jac='2-point', hess=BFGS()
        )
        
        ### Initial value is param0_high
        param0_high_np = np.array(param0_high)

        # First optimization attempt
        solution1_nobd = cobyqa.minimize(
            lambda x: objective0(x, Z_step), 
            param0_high_np,
            bounds=bounds0, 
            constraints=(constraint_conserve, constraint_revenue), 
            options={'disp': True, 'feasibility_tol': 1e-6, 'radius_init': 0.5, 'radius_final': 0.05}
        )

        solution1_nobd.x = np.array(solution1_nobd.x)

        # Retry if optimization did not converge
        if not solution1_nobd.success:
            print(f"Step {step}: First attempt did not converge. Retrying with new initial guess.")
            solution1_nobd_2 = cobyqa.minimize(
                lambda x: objective0(x, Z_step), 
                np.array(solution1_nobd.x, dtype=float),  # Ensure NumPy conversion
                bounds=bounds0, 
                constraints=(constraint_conserve, constraint_revenue), 
                options={'disp': True, 'feasibility_tol': 1e-6, 'radius_init': 0.5, 'radius_final': 0.05}
            )
            solution1_nobd_final = solution1_nobd_2
        else:
            print(f"Step {step}: Optimization converged successfully.")
            solution1_nobd_final = solution1_nobd

        # **Check for Constraint Violations**
        conserve_value = conservation_constraint_jitted(solution1_nobd_final.x, Z_step)
        revenue_value = revenue_lower_bound_constraint_jitted(solution1_nobd_final.x, Z_step)

        if conserve_value < 0.95 or conserve_value > 1.0:
            raise ValueError(f"Step {step}: Conservation constraint violated! Value: {conserve_value}")

        if revenue_value < 0.0:
            raise ValueError(f"Step {step}: Revenue constraint violated! Value: {revenue_value}")

        # Compute optimal price and quantities
        p_l, q_l, fc_l = param_to_pq0_jitted(solution1_nobd_final.x)

        # Process constraints
        processed_Z = average_Z_jitted(Z_step)
        q_sum_hh_step = get_q_sum_hh_jitted(p_l, q_l, fc_l, processed_Z)
        r_step = from_q_to_r_jitted(q_sum_hh_step, p_l, q_l, fc_l)
        r_step_agg = nansum_ignore_nan_inf_jitted(r_step) / 12
        cs_step = get_v_out_jitted(q_sum_hh_step, p_l, q_l, fc_l, processed_Z)
        cs_step_agg = nansum_ignore_nan_inf_jitted(cs_step) / 12
        q_hh_step = q_sum_hh_step/sim
        q_step_agg = nansum_ignore_nan_inf_jitted(q_hh_step) / 12

        # Append results
        pl_step_agg_results.append(p_l)
        ql_step_agg_results.append(q_l)
        fcl_step_agg_results.append(fc_l)
        r_step_agg_results.append(r_step_agg)
        cs_step_agg_results.append(cs_step_agg)
        q_step_agg_results.append(q_step_agg)
        cs_step_results.append(cs_step)
        q_step_results.append(q_hh_step)
        r_step_results.append(r_step)

        # Clean up memory
        del q_sum_hh_step, r_step, cs_step, q_hh_step
        gc.collect()

    except ValueError as e:
        error_log.append(str(e))  # Store error message
        print(f"Error at step {step}: {e}")  # Optional: Print errors immediately

# **Print all errors after the loop**
if error_log:
    print("\nErrors encountered during optimization:")
    for err in error_log:
        print(err)



"""
for step in steps:
    # Update Z_step with the current step
    Z_step = Z_history.copy()  # Make a copy of Z_history to preserve its original structure
    Z_step = Z_step.at[:, 8:12].add(step)  # Modify the selected slice (columns 8 to 11) by adding `step`
    Z_step = jnp.maximum(Z_step, 1e-16)
    
    constraint_conserve = NonlinearConstraint(
        lambda x: conservation_constraint_jitted(x, Z_step ), 
        0.95, 1.0, jac='2-point', hess=BFGS()
    )
    constraint_revenue = NonlinearConstraint(
        lambda x: revenue_lower_bound_constraint_jitted(x, Z_step ), 
        0.0, jnp.inf, jac='2-point', hess=BFGS()
    )

    #param0_2 = param0_2.astype(float)
    param0_2_np = np.array(param0_2)

    solution1_nobd = cobyqa.minimize(
        lambda x: objective0(x, Z_step ), 
        param0_2_np,
        bounds=bounds0, 
        constraints=(constraint_conserve, constraint_revenue), 
        options={'disp': True, 'feasibility_tol': 1e-9,'radius_init': 0.1, 'radius_final': 0.01}
    )

     #Check convergence and retry if necessary
    solution1_nobd.x = np.array(solution1_nobd.x)
    if not solution1_nobd.success:
        print("First attempt did not converge. Retrying with new initial guess.")
        solution1_nobd_2 = cobyqa.minimize(
            lambda x: objective0(x, Z_step ), 
            np.array(solution1_nobd.x, dtype=float),  # Ensure NumPy conversion
            bounds=bounds0, 
            constraints=(constraint_conserve, constraint_revenue), 
            options={'disp': True, 'feasibility_tol': 1e-9,'radius_init': 0.1, 'radius_final': 0.01}
        )
        solution1_nobd_final = solution1_nobd_2
    else:
        print("Optimization converged successfully.")
        solution1_nobd_final = solution1_nobd

    #solution1_nobd_final = solution1_nobd 
    p_l, q_l, fc_l = param_to_pq0_jitted(solution1_nobd_final.x)
    ## Compute optimal price
    #p_l, q_l, fc_l = loop_Z_get_price(Z_step)
    # Compute quantities
    ### Use Average at first
    processed_Z = average_Z_jitted(Z_step)
    q_sum_hh_step = get_q_sum_hh_jitted(p_l, q_l, fc_l, processed_Z)
    r_step = from_q_to_r_jitted(q_sum_hh_step, p_l, q_l, fc_l)
    #r_step_filtered =r_step[r_step < 20000]
    #r_step = r_step * (r_step < 20000)
    #r_step_agg = sum_ignore_outliers_jitted(r_step) / 12
    r_step_agg = nansum_ignore_nan_inf_jitted(r_step) / 12
    cs_step = get_v_out_jitted(q_sum_hh_step, p_l, q_l, fc_l, processed_Z)
    #cs_step_filtered = cs_step[cs_step > -0.5 * 1e9]
    #cs_step = cs_step * (cs_step > -0.5 * 1e9)
    #cs_step_agg = sum_ignore_outliers_jitted(cs_step) / 12
    cs_step_agg = nansum_ignore_nan_inf_jitted(cs_step) / 12
    del q_sum_hh_step, r_step
    
    # Append results
    pl_step_agg_results.append(p_l)
    ql_step_agg_results.append(q_l)
    fcl_step_agg_results.append(fc_l)
    r_step_agg_results.append(r_step_agg)
    cs_step_agg_results.append(cs_step_agg)
    cs_step_results.append(cs_step)
    del cs_step
    gc.collect()
"""
# Convert results to arrays for further processing
pl_step_agg_results = jnp.array(pl_step_agg_results)
pl_step_agg_results  = pl_step_agg_results.T
pl_step_agg_results_df = pd.DataFrame(pl_step_agg_results)
pl_step_agg_results_df.to_csv("ramsey_price_result/price_detail_results/current_info_avg_bound_mean_pl.csv", index=False)
del pl_step_agg_results, pl_step_agg_results_df

ql_step_agg_results = jnp.array(ql_step_agg_results)
ql_step_agg_results  = ql_step_agg_results.T
ql_step_agg_results_df = pd.DataFrame(ql_step_agg_results)
ql_step_agg_results_df.to_csv("ramsey_price_result/price_detail_results/current_info_avg_bound_mean_ql.csv", index=False)
del ql_step_agg_results, ql_step_agg_results_df

fcl_step_agg_results = jnp.array(fcl_step_agg_results)
fcl_step_agg_results  = fcl_step_agg_results.T
fcl_step_agg_results_df = pd.DataFrame(fcl_step_agg_results)
fcl_step_agg_results_df.to_csv("ramsey_price_result/price_detail_results/current_info_avg_bound_mean_fcl.csv", index=False)
del fcl_step_agg_results, fcl_step_agg_results_df

r_step_agg_results = jnp.array(r_step_agg_results)
r_step_agg_results_df = pd.DataFrame(r_step_agg_results)
r_step_agg_results_df.to_csv("ramsey_welfare_result/current_info_avg_bound_mean_r.csv", index=False)

cs_step_agg_results = jnp.array(cs_step_agg_results)
cs_step_agg_results_df = pd.DataFrame(cs_step_agg_results)
cs_step_agg_results_df.to_csv("ramsey_welfare_result/current_info_avg_bound_mean_cs.csv", index=False)

q_step_agg_results = jnp.array(q_step_agg_results)
q_step_agg_results_df = pd.DataFrame(q_step_agg_results)
q_step_agg_results_df.to_csv("ramsey_welfare_result/current_info_avg_bound_mean_q.csv", index=False)

cs_step_results = jnp.array(cs_step_results)
cs_step_results=cs_step_results.T
cs_step_results_df = pd.DataFrame(cs_step_results)
cs_step_results_df.to_csv("ramsey_welfare_result/cs_detail_results/current_info_avg_bound_mean_cs_steps.csv", index=False)
del cs_step_results, cs_step_results_df

q_step_results = jnp.array(q_step_results)
q_step_results=q_step_results.T
q_step_results_df = pd.DataFrame(q_step_results)
q_step_results_df.to_csv("ramsey_welfare_result/cs_detail_results/current_info_avg_bound_mean_q_steps.csv", index=False)
del q_step_results, q_step_results_df

r_step_results = jnp.array(r_step_results)
r_step_results=r_step_results.T
r_step_results_df = pd.DataFrame(r_step_results)
r_step_results_df.to_csv("ramsey_welfare_result/cs_detail_results/current_info_avg_bound_mean_r_steps.csv", index=False)
del r_step_results, r_step_results_df

#######################################
#### Prepare Z for changing IQR ######
######################################

#### noted that the iqr in the demand model is iqr within month. This is not the focus of the research
#### The research focus on the volatility across different month both within a year




# Function to compute adjusted values (pure function)
def adjust_min_max(arr, scale):
    # Sort indices for each column
    sorted_indices = jnp.argsort(arr, axis=0)

    # Get bottom 3 and top 3 indices
    bottom_3 = sorted_indices[:3, :]
    top_3 = sorted_indices[-3:, :]

    # Create a copy to avoid modifying the original array
    adjusted = arr.copy()

    # Decrease the bottom 3 values
    adjusted = adjusted.at[bottom_3, jnp.arange(arr.shape[1])].add(-scale)

    # Increase the top 3 values
    adjusted = adjusted.at[top_3, jnp.arange(arr.shape[1])].add(scale)

    return adjusted

# JIT the function
adjust_min_max_jitted = jax.jit(adjust_min_max)

def scale_sd(arr, scale_factor):
    """Scales standard deviation of arr and computes a summary statistic."""
    mean = jnp.mean(arr, axis=0, keepdims=True)
    #std = jnp.std(arr, axis=0, keepdims=True)
    
    # Normalize & scale standard deviation
    rescaled = mean + (arr - mean) * scale_factor
   # rescaled = jnp.maximum(rescaled, 0)  # Ensure non-negative values
    return rescaled

scale_sd_jitted = jax.jit(scale_sd)

s_steps = jnp.arange(0.75, 1.25+0.05, 0.05)

"""
### For extreme bound var

param0_high = jnp.array([5, 45, 45, 45, 45, 
                    2, 
                    6-2, 11-6, 20-11,
                    20, 
                    70, 70, 70
                    , 70
                    ])

### for small sd logr

param0_high = jnp.array([3, 4, 4, 4, 4, 
                    2, 
                    6-2, 11-6, 20-11,
                    8, 
                    7, 7, 7
                    , 7
                    ])

param0_high = jnp.array([3, 3.5, 3.5, 3.5, 3.5, 
                    2, 
                    6-2, 11-6, 20-11,
                    8, 
                    7, 7, 7
                    , 7
                    ])
"""
pl_step_agg_results_iqr = []
ql_step_agg_results_iqr = []
fcl_step_agg_results_iqr = []
r_step_agg_results_iqr = []
cs_step_agg_results_iqr = []
q_step_agg_results_iqr = []
cs_step_results_iqr = []
q_step_results_iqr = []
r_step_results_iqr = []
error_log_var = []

param0_low = jnp.array([3, 3, 3, 3, 3, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    7,7,7,7
                    ])

param0_high = jnp.array([3, 4, 4, 4, 4, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    7.125,7.125,7.125,7.125
                    ])


def get_initial_param_for_step(step, param_high, param_low):
    """
    Determines the initial parameter value based on the step according to specified ranges.
    """
    # Use a small tolerance for floating point comparisons
    tolerance = 1e-9

    if abs(step - 0.75) < tolerance or (step >= 1.1 - tolerance and step <= 1.25 + tolerance):
        print(f"Step {step}: Using param_high as initial guess.")
        return param_high
    else:
        print(f"Step {step}: Using param0_low (default) as initial guess.")
        return param_low
    
get_initial_param_for_step_jitted = jax.jit(get_initial_param_for_step)



for step in s_steps:
    try:
        # Update Z_step with the current step
        #Z_step = Z_history.copy()  # Preserve original structure
        Z_step = Z_current_duplicate.copy()  # Preserve original structure
        Z_step = Z_step.at[:, 8:12].set(scale_sd_jitted(Z_step[:, 8:12], step))
        Z_step = jnp.maximum(Z_step, 1e-16)

        # Define constraints
        constraint_conserve = NonlinearConstraint(
            lambda x: conservation_constraint_jitted(x, Z_step), 
            0.95, 1.0, jac='2-point', hess=BFGS()
        )
        constraint_revenue = NonlinearConstraint(
            lambda x: revenue_lower_bound_constraint_jitted(x, Z_step), 
            0.0, jnp.inf, jac='2-point', hess=BFGS()
        )
        
        ### Determine the initial value based on the current step
        # Call the function to get the initial parameter for this step
        initial_param_for_step = get_initial_param_for_step(step, param0_high, param0_low)
        param0_high_np = np.array(initial_param_for_step, dtype=float) # Ensure NumPy conversion

        # First optimization attempt
        solution1_nobd = cobyqa.minimize(
            lambda x: objective0(x, Z_step), 
            param0_high_np,
            bounds=bounds0, 
            constraints=(constraint_conserve, constraint_revenue), 
            options={'disp': True, 'feasibility_tol': 1e-6, 'radius_init': 0.5, 'radius_final': 0.05}
        )

        solution1_nobd.x = np.array(solution1_nobd.x)

        # Retry if optimization did not converge
        if not solution1_nobd.success:
            print(f"Step {step}: First attempt did not converge. Retrying with new initial guess.")
            solution1_nobd_2 = cobyqa.minimize(
                lambda x: objective0(x, Z_step), 
                np.array(solution1_nobd.x, dtype=float),  # Ensure NumPy conversion
                bounds=bounds0, 
                constraints=(constraint_conserve, constraint_revenue), 
                options={'disp': True, 'feasibility_tol': 1e-6, 'radius_init': 0.5, 'radius_final': 0.05}
            )
            solution1_nobd_final = solution1_nobd_2
        else:
            print(f"Step {step}: Optimization converged successfully.")
            solution1_nobd_final = solution1_nobd

        # **Check for Constraint Violations**
        conserve_value = conservation_constraint_jitted(solution1_nobd_final.x, Z_step)
        revenue_value = revenue_lower_bound_constraint_jitted(solution1_nobd_final.x, Z_step)

        if conserve_value < 0.95 or conserve_value > 1.0:
            raise ValueError(f"Step {step}: Conservation constraint violated! Value: {conserve_value}")

        if revenue_value < 0.0:
            raise ValueError(f"Step {step}: Revenue constraint violated! Value: {revenue_value}")

        # Compute optimal price and quantities
        p_l, q_l, fc_l = param_to_pq0_jitted(solution1_nobd_final.x)

        # Process constraints
        processed_Z = average_Z_jitted(Z_step)
        q_sum_hh_step = get_q_sum_hh_jitted(p_l, q_l, fc_l, processed_Z)
        r_step = from_q_to_r_jitted(q_sum_hh_step, p_l, q_l, fc_l)
        r_step_agg = nansum_ignore_nan_inf_jitted(r_step) / 12
        cs_step = get_v_out_jitted(q_sum_hh_step, p_l, q_l, fc_l, processed_Z)
        cs_step_agg = nansum_ignore_nan_inf_jitted(cs_step) / 12
        q_hh_step = q_sum_hh_step/sim
        q_step_agg = nansum_ignore_nan_inf_jitted(q_hh_step) / 12

        # Append results
        pl_step_agg_results_iqr.append(p_l)
        ql_step_agg_results_iqr.append(q_l)
        fcl_step_agg_results_iqr.append(fc_l)
        r_step_agg_results_iqr.append(r_step_agg)
        cs_step_agg_results_iqr.append(cs_step_agg)
        q_step_agg_results_iqr.append(q_step_agg)
        cs_step_results_iqr.append(cs_step)
        q_step_results_iqr.append(q_hh_step)
        r_step_results_iqr.append(r_step)

        # Clean up memory
        del q_sum_hh_step, r_step, cs_step, q_hh_step
        gc.collect()

    except ValueError as e:
        error_log_var.append(str(e))  # Store error message
        print(f"Error at step {step}: {e}")  # Optional: Print errors immediately

# **Print all errors after the loop**
if error_log_var:
    print("\nErrors encountered during optimization:")
    for err in error_log_var:
        print(err)
        
"""
p_l
q_l
fc_l
cs_step_agg
q_step_agg
r_step_agg
(cs_step_agg - cs_agg_0)/abs(cs_agg_0)
(q_step_agg - q_agg_0)/abs(q_agg_0)
(r_step_agg - r_agg_0)/abs(r_agg_0)

cs_r_q = jnp.column_stack((cs_step, r_step, q_hh_step))
# Convert to NumPy
cs_r_q = np.array(cs_r_q)

# Export to CSV
np.savetxt("ramsey_welfare_result/cs_detail_results/cs_r_q_gamma05_115.csv", cs_r_q, delimiter=",", fmt="%.5f")
"""        

# Convert results to arrays for further processing
pl_step_agg_results_iqr = jnp.array(pl_step_agg_results_iqr)
pl_step_agg_results_iqr  = pl_step_agg_results_iqr.T
pl_step_agg_results_iqr_df = pd.DataFrame(pl_step_agg_results_iqr)
pl_step_agg_results_iqr_df.to_csv("ramsey_price_result/price_detail_results/current_info_avg_bound_var_pl.csv", index=False)
del pl_step_agg_results_iqr, pl_step_agg_results_iqr_df

ql_step_agg_results_iqr = jnp.array(ql_step_agg_results_iqr)
ql_step_agg_results_iqr  = ql_step_agg_results_iqr.T
ql_step_agg_results_iqr_df = pd.DataFrame(ql_step_agg_results_iqr)
ql_step_agg_results_iqr_df.to_csv("ramsey_price_result/price_detail_results/current_info_avg_bound_var_ql.csv", index=False)
del ql_step_agg_results_iqr, ql_step_agg_results_iqr_df

fcl_step_agg_results_iqr = jnp.array(fcl_step_agg_results_iqr)
fcl_step_agg_results_iqr  = fcl_step_agg_results_iqr.T
fcl_step_agg_results_iqr_df = pd.DataFrame(fcl_step_agg_results_iqr)
fcl_step_agg_results_iqr_df.to_csv("ramsey_price_result/price_detail_results/current_info_avg_bound_var_fcl.csv", index=False)
del fcl_step_agg_results_iqr, fcl_step_agg_results_iqr_df

r_step_agg_results_iqr = jnp.array(r_step_agg_results_iqr)
r_step_agg_results_iqr_df = pd.DataFrame(r_step_agg_results_iqr)
r_step_agg_results_iqr_df.to_csv("ramsey_welfare_result/current_info_avg_bound_var_r.csv", index=False)

cs_step_agg_results_iqr = jnp.array(cs_step_agg_results_iqr)
cs_step_agg_results_iqr_df = pd.DataFrame(cs_step_agg_results_iqr)
cs_step_agg_results_iqr_df.to_csv("ramsey_welfare_result/current_info_avg_bound_var_cs.csv", index=False)

q_step_agg_results_iqr = jnp.array(q_step_agg_results_iqr)
q_step_agg_results_iqr_df = pd.DataFrame(q_step_agg_results_iqr)
q_step_agg_results_iqr_df.to_csv("ramsey_welfare_result/current_info_avg_bound_var_q.csv", index=False)

cs_step_results_iqr = jnp.array(cs_step_results_iqr)
cs_step_results_iqr=cs_step_results_iqr.T
cs_step_results_iqr_df = pd.DataFrame(cs_step_results_iqr)
cs_step_results_iqr_df.to_csv("ramsey_welfare_result/cs_detail_results/current_info_avg_bound_var_cs_steps.csv", index=False)
del cs_step_results_iqr, cs_step_results_iqr_df

q_step_results_iqr = jnp.array(q_step_results_iqr)
q_step_results_iqr=q_step_results_iqr.T
q_step_results_iqr_df = pd.DataFrame(q_step_results_iqr)
q_step_results_iqr_df.to_csv("ramsey_welfare_result/cs_detail_results/current_info_avg_bound_var_q_steps.csv", index=False)
del q_step_results_iqr, q_step_results_iqr_df

r_step_results_iqr = jnp.array(r_step_results_iqr)
r_step_results_iqr=r_step_results_iqr.T
r_step_results_iqr_df = pd.DataFrame(r_step_results_iqr)
r_step_results_iqr_df.to_csv("ramsey_welfare_result/cs_detail_results/current_info_avg_bound_var_r_steps.csv", index=False)
del r_step_results_iqr, r_step_results_iqr_df

# Convert results to DataFrame
results_df = pd.DataFrame({
    "steps": steps,
    "s_steps": s_steps,
    "mean_r": r_step_agg_results,
    "mean_cs": cs_step_agg_results,
    "mean_q": q_step_agg_results,
    "var_r": r_step_agg_results_iqr,
    "var_cs": cs_step_agg_results_iqr,
    "var_q": q_step_agg_results_iqr,
})

results_df['mean_r_diff'] = (results_df['mean_r'] -r_agg_0)/r_agg_0
results_df['var_r_diff'] = (results_df['var_r'] -r_agg_0)/r_agg_0
results_df['mean_q_diff'] = (results_df['mean_q'] -q_agg_0)/q_agg_0
results_df['var_q_diff'] = (results_df['var_q'] -q_agg_0)/q_agg_0
results_df['mean_cs_diff'] = (results_df['mean_cs'] -cs_agg_0)/np.absolute(cs_agg_0)
results_df['var_cs_diff'] = (results_df['var_cs'] -cs_agg_0)/np.absolute(cs_agg_0)

results_df.to_csv("ramsey_welfare_result/current_info_avg_bound_result.csv", index=False)

########################################
###### CRRA Function ##########
########################################

steps = jnp.arange(-0.25, 0.3, 0.05)

#s_steps = jnp.arange(-0.05,0.35-0.025 , 0.025)

# Lists to store results
pl_step_agg_results = []
ql_step_agg_results = []
fcl_step_agg_results = []
r_step_agg_results = []
cs_step_agg_results = []
q_step_agg_results = []
cs_step_results = []
q_step_results = []
r_step_results = []
error_log = []  # Store errors

param0_high = jnp.array([3, 7, 7, 7, 7, 
                    2, 
                    6-2, 11-6, 20-11,
                    8, 
                    18, 18, 18
                    , 18
                    ])

Z_current_duplicate = jnp.repeat(Z_current, repeats=4, axis=1)

for step in steps:
    try:
        # Update Z_step with the current step
        Z_step = Z_current_duplicate.copy()  # Preserve original structure
        #Z_step = Z_history.copy()  # Preserve original structure
        Z_step = Z_step.at[:, 8:12].add(step)  # Modify slice (columns 8 to 11)
        Z_step = jnp.maximum(Z_step, 1e-16)

        # Define constraints
        constraint_conserve = NonlinearConstraint(
            lambda x: conservation_constraint_jitted(x, Z_step), 
            0.95, 1.0, jac='2-point', hess=BFGS()
        )
        constraint_revenue = NonlinearConstraint(
            lambda x: revenue_lower_bound_crra_constraint_jitted(x, Z_step), 
            0.0, jnp.inf, jac='2-point', hess=BFGS()
        )
        
        ### Initial value is param0_high
        param0_high_np = np.array(param0_high)

        # First optimization attempt
        solution1_nobd = cobyqa.minimize(
            lambda x: objective0(x, Z_step), 
            param0_high_np,
            bounds=bounds0, 
            constraints=(constraint_conserve, constraint_revenue), 
            options={'disp': True, 'feasibility_tol': 1e-6, 'radius_init': 0.5, 'radius_final': 0.05}
        )

        solution1_nobd.x = np.array(solution1_nobd.x)

        # Retry if optimization did not converge
        if not solution1_nobd.success:
            print(f"Step {step}: First attempt did not converge. Retrying with new initial guess.")
            solution1_nobd_2 = cobyqa.minimize(
                lambda x: objective0(x, Z_step), 
                np.array(solution1_nobd.x, dtype=float),  # Ensure NumPy conversion
                bounds=bounds0, 
                constraints=(constraint_conserve, constraint_revenue), 
                options={'disp': True, 'feasibility_tol': 1e-6, 'radius_init': 0.5, 'radius_final': 0.05}
            )
            solution1_nobd_final = solution1_nobd_2
        else:
            print(f"Step {step}: Optimization converged successfully.")
            solution1_nobd_final = solution1_nobd

        # **Check for Constraint Violations**
        conserve_value = conservation_constraint_jitted(solution1_nobd_final.x, Z_step)
        revenue_value = revenue_lower_bound_crra_constraint_jitted(solution1_nobd_final.x, Z_step)

        if conserve_value < 0.95 or conserve_value > 1.0:
            raise ValueError(f"Step {step}: Conservation constraint violated! Value: {conserve_value}")

        if revenue_value < 0.0:
            raise ValueError(f"Step {step}: Revenue constraint violated! Value: {revenue_value}")

        # Compute optimal price and quantities
        p_l, q_l, fc_l = param_to_pq0_jitted(solution1_nobd_final.x)

        # Process constraints
        processed_Z = average_Z_jitted(Z_step)
        q_sum_hh_step = get_q_sum_hh_jitted(p_l, q_l, fc_l, processed_Z)
        r_step = from_q_to_r_jitted(q_sum_hh_step, p_l, q_l, fc_l)
        r_step_agg = nansum_ignore_nan_inf_jitted(r_step) / 12
        cs_step = get_v_out_jitted(q_sum_hh_step, p_l, q_l, fc_l, processed_Z)
        cs_step_agg = nansum_ignore_nan_inf_jitted(cs_step) / 12
        q_hh_step = q_sum_hh_step/sim
        q_step_agg = nansum_ignore_nan_inf_jitted(q_hh_step) / 12

        # Append results
        pl_step_agg_results.append(p_l)
        ql_step_agg_results.append(q_l)
        fcl_step_agg_results.append(fc_l)
        r_step_agg_results.append(r_step_agg)
        cs_step_agg_results.append(cs_step_agg)
        q_step_agg_results.append(q_step_agg)
        cs_step_results.append(cs_step)
        q_step_results.append(q_hh_step)
        r_step_results.append(r_step)

        # Clean up memory
        del q_sum_hh_step, r_step, cs_step, q_hh_step
        gc.collect()

    except ValueError as e:
        error_log.append(str(e))  # Store error message
        print(f"Error at step {step}: {e}")  # Optional: Print errors immediately

# **Print all errors after the loop**
if error_log:
    print("\nErrors encountered during optimization:")
    for err in error_log:
        print(err)



"""
for step in steps:
    # Update Z_step with the current step
    Z_step = Z_history.copy()  # Make a copy of Z_history to preserve its original structure
    Z_step = Z_step.at[:, 8:12].add(step)  # Modify the selected slice (columns 8 to 11) by adding `step`
    Z_step = jnp.maximum(Z_step, 1e-16)
    
    constraint_conserve = NonlinearConstraint(
        lambda x: conservation_constraint_jitted(x, Z_step ), 
        0.95, 1.0, jac='2-point', hess=BFGS()
    )
    constraint_revenue = NonlinearConstraint(
        lambda x: revenue_lower_bound_constraint_jitted(x, Z_step ), 
        0.0, jnp.inf, jac='2-point', hess=BFGS()
    )

    #param0_2 = param0_2.astype(float)
    param0_2_np = np.array(param0_2)

    solution1_nobd = cobyqa.minimize(
        lambda x: objective0(x, Z_step ), 
        param0_2_np,
        bounds=bounds0, 
        constraints=(constraint_conserve, constraint_revenue), 
        options={'disp': True, 'feasibility_tol': 1e-9,'radius_init': 0.1, 'radius_final': 0.01}
    )

     #Check convergence and retry if necessary
    solution1_nobd.x = np.array(solution1_nobd.x)
    if not solution1_nobd.success:
        print("First attempt did not converge. Retrying with new initial guess.")
        solution1_nobd_2 = cobyqa.minimize(
            lambda x: objective0(x, Z_step ), 
            np.array(solution1_nobd.x, dtype=float),  # Ensure NumPy conversion
            bounds=bounds0, 
            constraints=(constraint_conserve, constraint_revenue), 
            options={'disp': True, 'feasibility_tol': 1e-9,'radius_init': 0.1, 'radius_final': 0.01}
        )
        solution1_nobd_final = solution1_nobd_2
    else:
        print("Optimization converged successfully.")
        solution1_nobd_final = solution1_nobd

    #solution1_nobd_final = solution1_nobd 
    p_l, q_l, fc_l = param_to_pq0_jitted(solution1_nobd_final.x)
    ## Compute optimal price
    #p_l, q_l, fc_l = loop_Z_get_price(Z_step)
    # Compute quantities
    ### Use Average at first
    processed_Z = average_Z_jitted(Z_step)
    q_sum_hh_step = get_q_sum_hh_jitted(p_l, q_l, fc_l, processed_Z)
    r_step = from_q_to_r_jitted(q_sum_hh_step, p_l, q_l, fc_l)
    #r_step_filtered =r_step[r_step < 20000]
    #r_step = r_step * (r_step < 20000)
    #r_step_agg = sum_ignore_outliers_jitted(r_step) / 12
    r_step_agg = nansum_ignore_nan_inf_jitted(r_step) / 12
    cs_step = get_v_out_jitted(q_sum_hh_step, p_l, q_l, fc_l, processed_Z)
    #cs_step_filtered = cs_step[cs_step > -0.5 * 1e9]
    #cs_step = cs_step * (cs_step > -0.5 * 1e9)
    #cs_step_agg = sum_ignore_outliers_jitted(cs_step) / 12
    cs_step_agg = nansum_ignore_nan_inf_jitted(cs_step) / 12
    del q_sum_hh_step, r_step
    
    # Append results
    pl_step_agg_results.append(p_l)
    ql_step_agg_results.append(q_l)
    fcl_step_agg_results.append(fc_l)
    r_step_agg_results.append(r_step_agg)
    cs_step_agg_results.append(cs_step_agg)
    cs_step_results.append(cs_step)
    del cs_step
    gc.collect()
"""
# Convert results to arrays for further processing
pl_step_agg_results = jnp.array(pl_step_agg_results)
pl_step_agg_results  = pl_step_agg_results.T
pl_step_agg_results_df = pd.DataFrame(pl_step_agg_results)
pl_step_agg_results_df.to_csv("ramsey_price_result/price_detail_results/current_info_gamma1_bound_mean_pl.csv", index=False)
del pl_step_agg_results, pl_step_agg_results_df

ql_step_agg_results = jnp.array(ql_step_agg_results)
ql_step_agg_results  = ql_step_agg_results.T
ql_step_agg_results_df = pd.DataFrame(ql_step_agg_results)
ql_step_agg_results_df.to_csv("ramsey_price_result/price_detail_results/current_info_gamma1_bound_mean_ql.csv", index=False)
del ql_step_agg_results, ql_step_agg_results_df

fcl_step_agg_results = jnp.array(fcl_step_agg_results)
fcl_step_agg_results  = fcl_step_agg_results.T
fcl_step_agg_results_df = pd.DataFrame(fcl_step_agg_results)
fcl_step_agg_results_df.to_csv("ramsey_price_result/price_detail_results/current_info_gamma1_bound_mean_fcl.csv", index=False)
del fcl_step_agg_results, fcl_step_agg_results_df

r_step_agg_results = jnp.array(r_step_agg_results)
r_step_agg_results_df = pd.DataFrame(r_step_agg_results)
r_step_agg_results_df.to_csv("ramsey_welfare_result/current_info_gamma1_bound_mean_r.csv", index=False)

cs_step_agg_results = jnp.array(cs_step_agg_results)
cs_step_agg_results_df = pd.DataFrame(cs_step_agg_results)
cs_step_agg_results_df.to_csv("ramsey_welfare_result/current_info_gamma1_bound_mean_cs.csv", index=False)

q_step_agg_results = jnp.array(q_step_agg_results)
q_step_agg_results_df = pd.DataFrame(q_step_agg_results)
q_step_agg_results_df.to_csv("ramsey_welfare_result/current_info_gamma1_bound_mean_q.csv", index=False)

cs_step_results = jnp.array(cs_step_results)
cs_step_results=cs_step_results.T
cs_step_results_df = pd.DataFrame(cs_step_results)
cs_step_results_df.to_csv("ramsey_welfare_result/cs_detail_results/current_info_gamma1_bound_mean_cs_steps.csv", index=False)
del cs_step_results, cs_step_results_df

q_step_results = jnp.array(q_step_results)
q_step_results=q_step_results.T
q_step_results_df = pd.DataFrame(q_step_results)
q_step_results_df.to_csv("ramsey_welfare_result/cs_detail_results/current_info_gamma1_bound_mean_q_steps.csv", index=False)
del q_step_results, q_step_results_df

r_step_results = jnp.array(r_step_results)
r_step_results=r_step_results.T
r_step_results_df = pd.DataFrame(r_step_results)
r_step_results_df.to_csv("ramsey_welfare_result/cs_detail_results/current_info_gamma1_bound_mean_r_steps.csv", index=False)
del r_step_results, r_step_results_df

#######################################
#### Prepare Z for changing IQR ######
######################################

#### noted that the iqr in the demand model is iqr within month. This is not the focus of the research
#### The research focus on the volatility across different month both within a year



s_steps = jnp.arange(0.75, 1.25+0.05, 0.05)

param0_high = jnp.array([3, 4, 4, 4, 4, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5,
                    7, 7, 7, 7,
                    ])

param0_low = jnp.array([3, 3.5, 3.5, 3.5, 3.5, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5,
                    7, 7, 7, 7,
                    ])

def get_initial_param_for_step(step, param_high, param_low):
    """
    Determines the initial parameter value based on the step according to specified ranges.
    """
    # Use a small tolerance for floating point comparisons
    tolerance = 1e-9

    if step <= 1.2 + tolerance:
        print(f"Step {step}: Using param_med as initial guess.")
        return param_low
    else:
        print(f"Step {step}: Using param0_high (default) as initial guess.")
        return param_high
    
get_initial_param_for_step_jitted = jax.jit(get_initial_param_for_step)

pl_step_agg_results_iqr = []
ql_step_agg_results_iqr = []
fcl_step_agg_results_iqr = []
r_step_agg_results_iqr = []
cs_step_agg_results_iqr = []
q_step_agg_results_iqr = []
cs_step_results_iqr = []
q_step_results_iqr = []
r_step_results_iqr = []
error_log_var = []

for step in s_steps:
    try:
        # Update Z_step with the current step
        #Z_step = Z_history.copy()  # Preserve original structure
        Z_step = Z_current_duplicate.copy()  # Preserve original structure
        Z_step = Z_step.at[:, 8:12].set(scale_sd_jitted(Z_step[:, 8:12], step))
        Z_step = jnp.maximum(Z_step, 1e-16)

        # Define constraints
        constraint_conserve = NonlinearConstraint(
            lambda x: conservation_constraint_jitted(x, Z_step), 
            0.95, 1.0, jac='2-point', hess=BFGS()
        )
        constraint_revenue = NonlinearConstraint(
            lambda x: revenue_lower_bound_crra_constraint_jitted(x, Z_step), 
            0.0, jnp.inf, jac='2-point', hess=BFGS()
        )
        
        initial_param_for_step = get_initial_param_for_step(step, param0_high, param0_low)
        param0_high_np = np.array(initial_param_for_step, dtype=float) # Ensure NumPy conversion

        # First optimization attempt
        solution1_nobd = cobyqa.minimize(
            lambda x: objective0(x, Z_step), 
            param0_high_np,
            bounds=bounds0, 
            constraints=(constraint_conserve, constraint_revenue), 
            options={'disp': True, 'feasibility_tol': 1e-6, 'radius_init': 0.5, 'radius_final': 0.05}
        )

        solution1_nobd.x = np.array(solution1_nobd.x)

        # Retry if optimization did not converge
        if not solution1_nobd.success:
            print(f"Step {step}: First attempt did not converge. Retrying with new initial guess.")
            solution1_nobd_2 = cobyqa.minimize(
                lambda x: objective0(x, Z_step), 
                np.array(solution1_nobd.x, dtype=float),  # Ensure NumPy conversion
                bounds=bounds0, 
                constraints=(constraint_conserve, constraint_revenue), 
                options={'disp': True, 'feasibility_tol': 1e-6, 'radius_init': 0.5, 'radius_final': 0.05}
            )
            solution1_nobd_final = solution1_nobd_2
        else:
            print(f"Step {step}: Optimization converged successfully.")
            solution1_nobd_final = solution1_nobd

        # **Check for Constraint Violations**
        conserve_value = conservation_constraint_jitted(solution1_nobd_final.x, Z_step)
        revenue_value = revenue_lower_bound_crra_constraint_jitted(solution1_nobd_final.x, Z_step)

        if conserve_value < 0.95 or conserve_value > 1.0:
            raise ValueError(f"Step {step}: Conservation constraint violated! Value: {conserve_value}")

        if revenue_value < 0.0:
            raise ValueError(f"Step {step}: Revenue constraint violated! Value: {revenue_value}")

        # Compute optimal price and quantities
        p_l, q_l, fc_l = param_to_pq0_jitted(solution1_nobd_final.x)

        # Process constraints
        processed_Z = average_Z_jitted(Z_step)
        q_sum_hh_step = get_q_sum_hh_jitted(p_l, q_l, fc_l, processed_Z)
        r_step = from_q_to_r_jitted(q_sum_hh_step, p_l, q_l, fc_l)
        r_step_agg = nansum_ignore_nan_inf_jitted(r_step) / 12
        cs_step = get_v_out_jitted(q_sum_hh_step, p_l, q_l, fc_l, processed_Z)
        cs_step_agg = nansum_ignore_nan_inf_jitted(cs_step) / 12
        q_hh_step = q_sum_hh_step/sim
        q_step_agg = nansum_ignore_nan_inf_jitted(q_hh_step) / 12

        # Append results
        pl_step_agg_results_iqr.append(p_l)
        ql_step_agg_results_iqr.append(q_l)
        fcl_step_agg_results_iqr.append(fc_l)
        r_step_agg_results_iqr.append(r_step_agg)
        cs_step_agg_results_iqr.append(cs_step_agg)
        q_step_agg_results_iqr.append(q_step_agg)
        cs_step_results_iqr.append(cs_step)
        q_step_results_iqr.append(q_hh_step)
        r_step_results_iqr.append(r_step)

        # Clean up memory
        del q_sum_hh_step, r_step, cs_step, q_hh_step
        gc.collect()

    except ValueError as e:
        error_log_var.append(str(e))  # Store error message
        print(f"Error at step {step}: {e}")  # Optional: Print errors immediately

# **Print all errors after the loop**
if error_log_var:
    print("\nErrors encountered during optimization:")
    for err in error_log_var:
        print(err)
        
"""
p_l
q_l
fc_l
cs_step_agg
q_step_agg
r_step_agg
(cs_step_agg - cs_agg_0)/abs(cs_agg_0)
(q_step_agg - q_agg_0)/abs(q_agg_0)
(r_step_agg - r_agg_0)/abs(r_agg_0)

cs_r_q = jnp.column_stack((cs_step, r_step, q_hh_step))
# Convert to NumPy
cs_r_q = np.array(cs_r_q)

# Export to CSV
np.savetxt("ramsey_welfare_result/cs_detail_results/cs_r_q_gamma1_115.csv", cs_r_q, delimiter=",", fmt="%.5f")
"""        

# Convert results to arrays for further processing
pl_step_agg_results_iqr = jnp.array(pl_step_agg_results_iqr)
pl_step_agg_results_iqr  = pl_step_agg_results_iqr.T
pl_step_agg_results_iqr_df = pd.DataFrame(pl_step_agg_results_iqr)
pl_step_agg_results_iqr_df.to_csv("ramsey_price_result/price_detail_results/current_info_gamma1_bound_var_pl.csv", index=False)
del pl_step_agg_results_iqr, pl_step_agg_results_iqr_df

ql_step_agg_results_iqr = jnp.array(ql_step_agg_results_iqr)
ql_step_agg_results_iqr  = ql_step_agg_results_iqr.T
ql_step_agg_results_iqr_df = pd.DataFrame(ql_step_agg_results_iqr)
ql_step_agg_results_iqr_df.to_csv("ramsey_price_result/price_detail_results/current_info_gamma1_bound_var_ql.csv", index=False)
del ql_step_agg_results_iqr, ql_step_agg_results_iqr_df

fcl_step_agg_results_iqr = jnp.array(fcl_step_agg_results_iqr)
fcl_step_agg_results_iqr  = fcl_step_agg_results_iqr.T
fcl_step_agg_results_iqr_df = pd.DataFrame(fcl_step_agg_results_iqr)
fcl_step_agg_results_iqr_df.to_csv("ramsey_price_result/price_detail_results/current_info_gamma1_bound_var_fcl.csv", index=False)
del fcl_step_agg_results_iqr, fcl_step_agg_results_iqr_df

r_step_agg_results_iqr = jnp.array(r_step_agg_results_iqr)
r_step_agg_results_iqr_df = pd.DataFrame(r_step_agg_results_iqr)
r_step_agg_results_iqr_df.to_csv("ramsey_welfare_result/current_info_gamma1_bound_var_r.csv", index=False)

cs_step_agg_results_iqr = jnp.array(cs_step_agg_results_iqr)
cs_step_agg_results_iqr_df = pd.DataFrame(cs_step_agg_results_iqr)
cs_step_agg_results_iqr_df.to_csv("ramsey_welfare_result/current_info_gamma1_bound_var_cs.csv", index=False)

q_step_agg_results_iqr = jnp.array(q_step_agg_results_iqr)
q_step_agg_results_iqr_df = pd.DataFrame(q_step_agg_results_iqr)
q_step_agg_results_iqr_df.to_csv("ramsey_welfare_result/current_info_gamma1_bound_var_q.csv", index=False)

cs_step_results_iqr = jnp.array(cs_step_results_iqr)
cs_step_results_iqr=cs_step_results_iqr.T
cs_step_results_iqr_df = pd.DataFrame(cs_step_results_iqr)
cs_step_results_iqr_df.to_csv("ramsey_welfare_result/cs_detail_results/current_info_gamma1_bound_var_cs_steps.csv", index=False)
del cs_step_results_iqr, cs_step_results_iqr_df

q_step_results_iqr = jnp.array(q_step_results_iqr)
q_step_results_iqr=q_step_results_iqr.T
q_step_results_iqr_df = pd.DataFrame(q_step_results_iqr)
q_step_results_iqr_df.to_csv("ramsey_welfare_result/cs_detail_results/current_info_gamma1_bound_var_q_steps.csv", index=False)
del q_step_results_iqr, q_step_results_iqr_df

r_step_results_iqr = jnp.array(r_step_results_iqr)
r_step_results_iqr=r_step_results_iqr.T
r_step_results_iqr_df = pd.DataFrame(r_step_results_iqr)
r_step_results_iqr_df.to_csv("ramsey_welfare_result/cs_detail_results/current_info_gamma1_bound_var_r_steps.csv", index=False)
del r_step_results_iqr, r_step_results_iqr_df

# Convert results to DataFrame
results_df = pd.DataFrame({
    "steps": steps,
    "s_steps": s_steps,
    "mean_r": r_step_agg_results,
    "mean_cs": cs_step_agg_results,
    "mean_q": q_step_agg_results,
    "var_r": r_step_agg_results_iqr,
    "var_cs": cs_step_agg_results_iqr,
    "var_q": q_step_agg_results_iqr,
})

results_df['mean_r_diff'] = (results_df['mean_r'] -r_agg_0)/r_agg_0
results_df['var_r_diff'] = (results_df['var_r'] -r_agg_0)/r_agg_0
results_df['mean_q_diff'] = (results_df['mean_q'] -q_agg_0)/q_agg_0
results_df['var_q_diff'] = (results_df['var_q'] -q_agg_0)/q_agg_0
results_df['mean_cs_diff'] = (results_df['mean_cs'] -cs_agg_0)/np.absolute(cs_agg_0)
results_df['var_cs_diff'] = (results_df['var_cs'] -cs_agg_0)/np.absolute(cs_agg_0)

results_df.to_csv("ramsey_welfare_result/current_info_gamma1_bound_result.csv", index=False)

#### Save Price Results
'''
solution1 = solution1_nobd_final
p_l1, q_l1, fc_l1 = param_to_pq0_jitted(solution1.x)
p_v1 = jnp.concatenate((p_l1, q_l1, fc_l1))

solution6 = solution1_nobd_final
p_l6, q_l6, fc_l6 = param_to_pq0_jitted(solution6.x)
p_v6 = jnp.concatenate((p_l6, q_l6, fc_l6))

solution8 = solution1_nobd_final
p_l8, q_l8, fc_l8 = param_to_pq0_jitted(solution8.x)
p_v8 = jnp.concatenate((p_l8, q_l8, fc_l8))

base_price_result = jnp.column_stack((p_v1, p_v6, p_v8))

np.savetxt("ramsey_price_result/ramsey_result_baseline.csv", base_price_result , delimiter=",", header="Model0,Model1,Model1.5", comments="", fmt="%.5f")

solution9 = solution1_nobd_final
p_l9, q_l9, fc_l9 = param_to_pq0_jitted(solution9.x)
p_v9 = jnp.concatenate((p_l9, q_l9, fc_l9))

solution10 = solution1_nobd_final
p_l10, q_l10, fc_l10 = param_to_pq0_jitted(solution10.x)
p_v10 = jnp.concatenate((p_l10, q_l10, fc_l10))

solution11 = solution1_nobd_final
p_l11, q_l11, fc_l11 = param_to_pq0_jitted(solution11.x)
p_v11 = jnp.concatenate((p_l11, q_l11, fc_l11))

conservation_price_result = jnp.column_stack((p_v9, p_v10, p_v11))

np.savetxt("ramsey_price_result/ramsey_result_conservation.csv", conservation_price_result , delimiter=",", header="Model0,Model1,Model1.5", comments="", fmt="%.5f")

'''

"""

solution125 = cobyqa.minimize(lambda x: objective(x, 1.25), 
                           param0, 
                           bounds = bounds, constraints=(constraint2), options={'disp': True,
                                                                            'feasibility_tol': 0.01, 
                                                                            #'radius_init':0.5, 
                                                                        'radius_final':0.01
                                                                        })

p_l, q_l, fc_l = param_to_pq_jitted(solution125.x)

q_sum_hh_low = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_low)
r_low125 = from_q_to_r_jitted(q_sum_hh_low, p_l, q_l, fc_l)
r_low125_agg = sum_ignore_outliers_jitted(r_low125)/12
cs_low125 = get_v_out_jitted(q_sum_hh_low, p_l, q_l, fc_l)
cs_low125_agg = sum_ignore_outliers_jitted(cs_low125)/12

q_sum_hh_high = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_high)
r_high125 = from_q_to_r_jitted(q_sum_hh_high, p_l, q_l, fc_l)
r_high125_agg = sum_ignore_outliers_jitted(r_high125)/12
cs_high125 = get_v_out_jitted(q_sum_hh_high, p_l, q_l, fc_l)
cs_high125_agg = sum_ignore_outliers_jitted(cs_high125)/12

del r_low125, cs_low125, r_high125, cs_high125

### No bounds on parameters. 
solution125_nobd = cobyqa.minimize(lambda x: objective0(x, 1.25), 
                           param0_2, 
                           bounds = bounds0, constraints=(constraint2), options={'disp': True,
                                                                            'feasibility_tol': 0.01, 
                                                                            #'radius_init':0.5, 
                                                                        'radius_final':0.01
                                                                        })
p_l, q_l, fc_l = param_to_pq_jitted(solution125_nobd.x)

q_sum_hh_low = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_low)
r_low125_nobd = from_q_to_r_jitted(q_sum_hh_low, p_l, q_l, fc_l)
r_low125_nobd_agg = sum_ignore_outliers_jitted(r_low125_nobd)/12
cs_low125_nobd = get_v_out_jitted(q_sum_hh_low, p_l, q_l, fc_l)
cs_low125_nobd_agg = sum_ignore_outliers_jitted(cs_low125_nobd)/12

q_sum_hh_high = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_high)
r_high125_nobd = from_q_to_r_jitted(q_sum_hh_high, p_l, q_l, fc_l)
r_high125_nobd_agg = sum_ignore_outliers_jitted(r_high125_nobd)/12
cs_high125_nobd = get_v_out_jitted(q_sum_hh_high, p_l, q_l, fc_l)
cs_high125_nobd_agg = sum_ignore_outliers_jitted(cs_high125_nobd)/12

del r_low125_nobd, cs_low125_nobd, r_high125_nobd, cs_high125_nobd

solution075 = cobyqa.minimize(lambda x: objective(x, 0.75), 
                           param0, 
                           bounds = bounds, constraints=(constraint2), options={'disp': True,
                                                                            'feasibility_tol': 0.01, 
                                                                            #'radius_init':0.5, 
                                                                        'radius_final':0.01
                                                                        })

p_l, q_l, fc_l = param_to_pq_jitted(solution075.x)

q_sum_hh_low = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_low)
r_low075 = from_q_to_r_jitted(q_sum_hh_low, p_l, q_l, fc_l)
r_low075_agg = sum_ignore_outliers_jitted(r_low075)/12
cs_low075 = get_v_out_jitted(q_sum_hh_low, p_l, q_l, fc_l)
cs_low075_agg = sum_ignore_outliers_jitted(cs_low075)/12

q_sum_hh_high = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_high)
r_high075 = from_q_to_r_jitted(q_sum_hh_high, p_l, q_l, fc_l)
r_high075_agg = sum_ignore_outliers_jitted(r_high075)/12
cs_high075 = get_v_out_jitted(q_sum_hh_high, p_l, q_l, fc_l)
cs_high075_agg = sum_ignore_outliers_jitted(cs_high075)/12

del r_low075, cs_low075, r_high075, cs_high075

### No bounds on parameters. 
solution075_nobd = cobyqa.minimize(lambda x: objective0(x, 0.75), 
                           param0_2, 
                           bounds = bounds0, constraints=(constraint2), options={'disp': True,
                                                                            'feasibility_tol': 0.01, 
                                                                            #'radius_init':0.5, 
                                                                        'radius_final':0.01
                                                                        })

p_l, q_l, fc_l = param_to_pq_jitted(solution075_nobd.x)

q_sum_hh_low = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_low)
r_low075_nobd = from_q_to_r_jitted(q_sum_hh_low, p_l, q_l, fc_l)
r_low075_nobd_agg = sum_ignore_outliers_jitted(r_low075_nobd)/12
cs_low075_nobd = get_v_out_jitted(q_sum_hh_low, p_l, q_l, fc_l)
cs_low075_nobd_agg = sum_ignore_outliers_jitted(cs_low075_nobd)/12

q_sum_hh_high = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_high)
r_high075_nobd = from_q_to_r_jitted(q_sum_hh_high, p_l, q_l, fc_l)
r_high075_nobd_agg = sum_ignore_outliers_jitted(r_high075_nobd)/12
cs_high075_nobd = get_v_out_jitted(q_sum_hh_high, p_l, q_l, fc_l)
cs_high075_nobd_agg = sum_ignore_outliers_jitted(cs_high075_nobd)/12

del r_low075_nobd, cs_low075_nobd, r_high075_nobd, cs_high075_nobd

solution15 = cobyqa.minimize(lambda x: objective(x, 1.5), 
                           param0, 
                           bounds = bounds, constraints=(constraint2), options={'disp': True,
                                                                            'feasibility_tol': 0.01, 
                                                                            #'radius_init':0.5, 
                                                                        'radius_final':0.01
                                                                        })

p_l, q_l, fc_l = param_to_pq_jitted(solution15.x)

q_sum_hh_low = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_low)
r_low15 = from_q_to_r_jitted(q_sum_hh_low, p_l, q_l, fc_l)
r_low15_agg = sum_ignore_outliers_jitted(r_low15)/12
cs_low15 = get_v_out_jitted(q_sum_hh_low, p_l, q_l, fc_l)
cs_low15_agg = sum_ignore_outliers_jitted(cs_low15)/12

q_sum_hh_high = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_high)
r_high15 = from_q_to_r_jitted(q_sum_hh_high, p_l, q_l, fc_l)
r_high15_agg = sum_ignore_outliers_jitted(r_high15)/12
cs_high15 = get_v_out_jitted(q_sum_hh_high, p_l, q_l, fc_l)
cs_high15_agg = sum_ignore_outliers_jitted(cs_high15)/12

del r_low15, cs_low15, r_high15, cs_high15

### No bounds on parameters. 
solution15_nobd = cobyqa.minimize(lambda x: objective0(x, 1.5), 
                           param0_2, 
                           bounds = bounds0, constraints=(constraint2), options={'disp': True,
                                                                            'feasibility_tol': 0.01, 
                                                                            #'radius_init':0.5, 
                                                                        'radius_final':0.01
                                                                        })

p_l, q_l, fc_l = param_to_pq_jitted(solution15_nobd.x)

q_sum_hh_low = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_low)
r_low15_nobd = from_q_to_r_jitted(q_sum_hh_low, p_l, q_l, fc_l)
r_low15_nobd_agg = sum_ignore_outliers_jitted(r_low15_nobd)/12
cs_low15_nobd = get_v_out_jitted(q_sum_hh_low, p_l, q_l, fc_l)
cs_low15_nobd_agg = sum_ignore_outliers_jitted(cs_low15_nobd)/12

q_sum_hh_high = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_high)
r_high15_nobd = from_q_to_r_jitted(q_sum_hh_high, p_l, q_l, fc_l)
r_high15_nobd_agg = sum_ignore_outliers_jitted(r_high15_nobd)/12
cs_high15_nobd = get_v_out_jitted(q_sum_hh_high, p_l, q_l, fc_l)
cs_high15_nobd_agg = sum_ignore_outliers_jitted(cs_high15_nobd)/12

del r_low15_nobd, cs_low15_nobd, r_high15_nobd, cs_high15_nobd

solution05 = cobyqa.minimize(lambda x: objective(x, 0.5), 
                           param0, 
                           bounds = bounds, constraints=(constraint2), options={'disp': True,
                                                                            'feasibility_tol': 0.01, 
                                                                            #'radius_init':0.5, 
                                                                        'radius_final':0.01
                                                                        })

p_l, q_l, fc_l = param_to_pq_jitted(solution05.x)

q_sum_hh_low = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_low)
r_low05 = from_q_to_r_jitted(q_sum_hh_low, p_l, q_l, fc_l)
r_low05_agg = sum_ignore_outliers_jitted(r_low05)/12
cs_low05 = get_v_out_jitted(q_sum_hh_low, p_l, q_l, fc_l)
cs_low05_agg = sum_ignore_outliers_jitted(cs_low05)/12

q_sum_hh_high = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_high)
r_high05 = from_q_to_r_jitted(q_sum_hh_high, p_l, q_l, fc_l)
r_high05_agg = sum_ignore_outliers_jitted(r_high05)/12
cs_high05 = get_v_out_jitted(q_sum_hh_high, p_l, q_l, fc_l)
cs_high05_agg = sum_ignore_outliers_jitted(cs_high05)/12

del r_low05, cs_low05, r_high05, cs_high05

### No bounds on parameters. 
solution05_nobd = cobyqa.minimize(lambda x: objective0(x, 0.5), 
                           param0_2, 
                           bounds = bounds0, constraints=(constraint2), options={'disp': True,
                                                                            'feasibility_tol': 0.01, 
                                                                            #'radius_init':0.5, 
                                                                        'radius_final':0.01
                                                                        })

p_l, q_l, fc_l = param_to_pq_jitted(solution05_nobd.x)

q_sum_hh_low = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_low)
r_low05_nobd = from_q_to_r_jitted(q_sum_hh_low, p_l, q_l, fc_l)
r_low05_nobd_agg = sum_ignore_outliers_jitted(r_low05_nobd)/12
cs_low05_nobd = get_v_out_jitted(q_sum_hh_low, p_l, q_l, fc_l)
cs_low05_nobd_agg = sum_ignore_outliers_jitted(cs_low05_nobd)/12

q_sum_hh_high = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_high)
r_high05_nobd = from_q_to_r_jitted(q_sum_hh_high, p_l, q_l, fc_l)
r_high05_nobd_agg = sum_ignore_outliers_jitted(r_high05_nobd)/12
cs_high05_nobd = get_v_out_jitted(q_sum_hh_high, p_l, q_l, fc_l)
cs_high05_nobd_agg = sum_ignore_outliers_jitted(cs_high05_nobd)/12

del r_low05_nobd, cs_low05_nobd, r_high05_nobd, cs_high05_nobd

###################################################
#### Find the optimal lambda, No constraints, ####
###################################################

def get_result_lam (p_l, q_l, fc_l, Z, lam):
    q_sum_hh = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z)
    r = from_q_to_r_jitted(q_sum_hh, p_l, q_l, fc_l)
    cs = get_v_out(q_sum_hh, p_l, q_l, fc_l)
    #result = 1 / (1+lam) * cs + lam / (1 + lam) * r
    result = cs + lam * r
    return result

get_result_lam_jitted = jax.jit(get_result_lam)

def objective_lam(param):
    #param = jnp.maximum(param, 0.01)
    jax.debug.print("Current param {p}", p= jax.device_get(param))
    lam = param[12]
    param = param[:12]
    p_l, q_l, fc_l = param_to_pq_jitted(param)
    result_low = get_result_lam_jitted(p_l, q_l, fc_l, Z_low, lam)
    result_high = get_result_lam_jitted(p_l, q_l, fc_l, Z_high, lam)
    result = (result_low + result_high)/2
    result = -1 * sum_ignore_outliers_jitted(result)
    result_value = jax.device_get(result)
    jax.debug.print("Current Value {r}", r= result_value)
    return result

objective_lam_jitted = jax.jit(objective_lam)

bounds_lam = Bounds([0, 0, 0, 0, 0, 
                 0, 0, 0,
                0, 0, 0,
                0,
                0
                ], 
                [
                 20, 20, 20, 20, 20, 
                 4, 9, 14,
                 20, 20, 20, 
                 20,
                 5
                 ])

param0_lam = jnp.array([3.09, 5.01-3.09, 8.54-5.01, 12.9-8.54, 14.41-12.9, 
                   # 2, 
                    6-2, 11-6, 20-11,
                    #8.5, 
                    10.8-8.5, 16.5-10.8, 37-16.5
                    , 37-37
                    ,0.5
                    ])

solutionlam = cobyqa.minimize(objective_lam, 
                           param0_lam, 
                           bounds = bounds_lam, 
                           #constraints=(constraint2), 
                           options={'disp': True,
                            'feasibility_tol': 0.01, 
                           #'radius_init':0.5, 
                           'radius_final':0.01
                            })

#solutionlam = minimize(objective_lam_jitted, param0_lam,bounds = bounds_lam,  method = 'Nelder-Mead', options={'maxfev': 20000})


def objective0_lam(param):
    param = jnp.maximum(param, 0.01)
    jax.debug.print("Current param {p}", p= jax.device_get(param))
    lam = param[14]
    param = param[:14]
    p_l, q_l, fc_l = param_to_pq0_jitted(param)
    result_low = get_result_lam_jitted(p_l, q_l, fc_l, Z_low, lam)
    result_high = get_result_lam_jitted(p_l, q_l, fc_l, Z_high, lam)
    result = (result_low + result_high)/2
    result = -1 * sum_ignore_outliers_jitted(result)
    result_value = jax.device_get(result)
    jax.debug.print("Current Value {r}", r= result_value)
    return result

objective0_lam_jitted = jax.jit(objective0_lam)

bounds0_lam = Bounds([0.01, 0.01, 0.01, 0.01, 0.01, 
                      0.01,
                 0.01, 0.01, 0.01,
                 0.01,
                0.01, 0.01, 0.01,
                0.01,
                0.5
                ], 
                [
                 jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, 
                                       jnp.inf,
                                  jnp.inf, jnp.inf, jnp.inf,
                                  jnp.inf,
                                 jnp.inf, jnp.inf, jnp.inf,
                                 jnp.inf,
                 2.5
                 ])

param0_2_lam = jnp.array([3.09, 5.01-3.09, 8.54-5.01, 12.9-8.54, 14.41-12.9, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    10.8-8.5, 16.5-10.8, 37-16.5
                    , 37-37,
                    0.75
                    ])

solutionlam_nobd = cobyqa.minimize(objective0_lam, 
                           param0_2_lam, 
                           bounds = bounds0_lam, 
                           constraints=(constraint2), 
                           options={'disp': True,
                                                                            'feasibility_tol': 0.01, 
                                                                            #'radius_init':0.5, 
                                                                        'radius_final':0.01
                                                                        })





# Prepare the storage for solution.x arrays
#solutions = []
#welfares = []

#lambda_values = np.arange(0.5, 2, 0.25)

#for lambda_ in lambda_values:
    # Run the COBYQA minimization with current lambda
 #   print(f"Current lambda: {lambda_}")
  #  result = minimize(lambda x: objective(x, lambda_), param0)

    # Store solution.x from the result if the minimization succeeded
   # if result.success:
    #    solutions.append(result.x)
    #    welfares.append(-1*result.fun)
    #else:
     #   print(f"Minimization failed for lambda = {lambda_}")

#solutions = np.array(solutions)
















######################
#### Revenue risks#####
########################

def revenue_diff(r):
    return nansum_ignore_nan_inf_jitted(jnp.square((r - r0)) / r0)/hh_size
 
revenue_diff_jitted = jax.jit(revenue_diff)

q0_sum_mean = jnp.mean(q0_sum)

######################
#### Conservation #####
########################

def cf_w_ci(q_sum, p_l, q_l, fc_l, q0_sum_mean = q0_sum_mean):
    #log_q = cf_w(p_l, q_l, fc_l,sigma_eta_df)
    #q = jnp.exp(log_q)
    #q0 = jnp.exp(log_q0)
    #q_sum = nansum_ignore_nan_inf_jitted(q)
    #q0_sums = jnp.nansum(q0, axis = 0)
    condition= (q0_sum_mean - q_sum)>0
    #condition= (0.75*q0_sum - q_sum) >0
    #count_satisfying_condition_p = jnp.count_nonzero(condition)/len(q_sum)
    return jnp.count_nonzero(condition)
cf_w_ci_jitted = jax.jit(cf_w_ci)

def conservation_condition(q_sum_sim, p_l, q_l, fc_l):
    conditions = cf_w_ci_jitted(q_sum_sim, p_l, q_l, fc_l)
    return conditions/len(q0_sum)
conservation_condition_jitted = jax.jit(conservation_condition)


######################
#### Equity   #####
########################

essential_q = demand_2018_using_new['essential_usage']

def get_new_p0(p_l, q_l):
    conditions_k = [
        (essential_q<q_l[0]),
        (( essential_q >=q_l[0]) & (essential_q < q_l[1])),
        (( essential_q >=q_l[1]) & (essential_q < q_l[2])),
        (( essential_q >=q_l[2]) & (essential_q < q_l[3])),
        (essential_q >= q_l[3]),
    ]  
    choices = [
        0,
        1,
        2,
        3,
        4
    ]
    result = jnp.select(conditions_k, choices)
    return p_l[result]
get_new_p0_jitted = jax.jit(get_new_p0)

def get_k(q_sum_hh, q_l):
    q_mean = q_sum_hh / sim
    conditions_k = [
        (q_mean<q_l[0]),
        (( q_mean >=q_l[0]) & (q_mean < q_l[1])), 
        (( q_mean >=q_l[1]) & (q_mean < q_l[2])),
        (( q_mean >=q_l[2]) & (q_mean < q_l[3])),
        (q_mean >= q_l[3]),
    ]  
    choices = [
        0,
        1,
        2,
        3,
        4
    ]
    result = jnp.select(conditions_k, choices)
    return result
get_k_jitted = jax.jit(get_k)

def get_virtual_income(q_sum_hh, p_l, q_l, fc_l):
    q_kink_l =q_l
    p_plus1_l = jnp.append(p_l[1:5],jnp.array([jnp.nan]) )
    d_end = jnp.cumsum( (p_l - p_plus1_l)[:4] *q_kink_l)
    d_end =  jnp.insert(d_end, 0, jnp.array([0.0]) )
    def calculate_dk (k):
        result = -fc_l[k] - d_end[k]
        return result
    calculate_dk_jitted = jax.jit(calculate_dk)
    
    k = get_k_jitted(q_sum_hh, q_l)
    virtual_income = jnp.maximum(jnp.multiply(jnp.transpose(calculate_dk_jitted(k)), de) + I, 1e-16)
    return virtual_income
get_virtual_income_jitted = jax.jit(get_virtual_income)

#def get_v_in(p_l, q_l, fc_l, I = I):
 #   new_p0 = get_new_p0_jitted(p_l, q_l)
  #  v_in = jnp.multiply(-1*jnp.exp(jnp.dot(A_current_indoor, b8) 
   #                + jnp.dot(Z_current_indoor, b9)
    #               + c_i), new_p0) + I
    #return v_in
#get_v_in_jitted = jax.jit(get_v_in)

alpha = jnp.exp(jnp.dot(A_current, b4)
                     + c_alpha)

def get_current_marginal_p(q_sum_hh, p_l, q_l, fc_l):
    k = get_k_jitted(q_sum_hh, q_l)
    p = p_l[k]
    return p
get_current_marginal_p_jitted = jax.jit(get_current_marginal_p)

def get_expenditure_in_v_out(q_sum_hh, p_l, q_l, fc_l):
    p = get_current_marginal_p_jitted(q_sum_hh, p_l, q_l, fc_l)
    result = jnp.multiply(jnp.exp(jnp.dot(A_current_outdoor, b1) + jnp.dot(Z_current, b2)), 
                                         jnp.divide(jnp.power(p, 1-alpha), jnp.array(1-alpha)))
    return result
get_expenditure_in_v_out_jitted = jax.jit(get_expenditure_in_v_out)

def get_v_out(q_sum_hh, p_l, q_l, fc_l):
    exp_v = get_expenditure_in_v_out_jitted(q_sum_hh, p_l,q_l, fc_l)
    sim_result_Ik = get_virtual_income_jitted(q_sum_hh, p_l, q_l, fc_l)
    v_out = -1 *exp_v  + sim_result_Ik ** (1-r) / (1-r)
    return v_out
get_v_out_jitted = jax.jit(get_v_out)

conditions_qsq = [
        (q_l0[0] <q_statusquo),
        ( (q_l0[0]>= q_statusquo) & (q_l0[1]< q_statusquo)),
        ( (q_l0[1]>= q_statusquo) & (q_l0[2]< q_statusquo)),
        ( (q_l0[2]>= q_statusquo) & (q_l0[3]< q_statusquo)),
        (  q_l0[3]>= q_statusquo),
    ]  
choices_psq = [
        p_l0[0],
        p_l0[1],
        p_l0[2],
        p_l0[3],
        p_l0[4]
    ]
result_psq = jnp.select(conditions_qsq, choices_psq)

def get_e_new_v_p0(q_sum_hh, p_l, q_l, fc_l):
    #k = get_k_jitted(q_sum_hh, q_l)
    #p = p_l[k]
    #v_in = get_v_in_jitted(q_sum_hh, p_l, q_l, fc_l)
    v_out = get_v_out_jitted(q_sum_hh, p_l, q_l, fc_l)
    
    e_out = (1-r)*jnp.power((v_out - jnp.multiply(jnp.exp(jnp.dot(A_current_outdoor, b1) + jnp.dot(Z_current_outdoor, b2)), 
                                    jnp.divide(jnp.power(result_psq, 1-alpha), 1-alpha ))), 1/(1-r))
    e_out  = jnp.multiply(e_out, de)
    return e_out
get_e_new_v_p0_jitted = jax.jit(get_e_new_v_p0)


def get_diff_payment(q_sum_hh, p_l, q_l, fc_l):
    k = get_k_jitted(q_sum_hh, q_l)
    q_l = jnp.insert(q_l, 0, 0)
    diff_payment = jnp.cumsum(jnp.multiply((p_l - p_l0), q_l))
    diff_payment = diff_payment[k]
    diff_payment = jnp.multiply(diff_payment, de)
    return diff_payment
get_diff_payment_jitted = jax.jit(get_diff_payment)
    
    
def get_ev(q_sum_hh, p_l, q_l, fc_l, I = I, de = de):
    e = get_e_new_v_p0_jitted(q_sum_hh, p_l, q_l, fc_l)
    diff_payment = get_diff_payment_jitted(q_sum_hh, p_l, q_l, fc_l)
    ev = e - diff_payment - I
    return ev
get_ev_jitted = jax.jit(get_ev)

def get_ev_perct_e(q_sum_hh, p_l, q_l, fc_l):
    ev=get_ev_jitted(q_sum_hh, p_l, q_l, fc_l)
    #r_mean = from_q_to_r_mean_jitted(q_sum_hh, p_l, q_l, fc_l)
    return jnp.divide(ev, I)
get_ev_perct_e_jitted = jax.jit(get_ev_perct_e)

def get_iqr(arr):
    #arr = arr[~jnp.isnan(arr) & ~jnp.isinf(arr)]
    mask = jnp.logical_and(jnp.isfinite(arr), ~jnp.isnan(arr))  # Mask out inf and NaN
    arr = jnp.where(mask, arr, 0)
    Q1 = jnp.percentile(arr, 25)  # 25th percentile
    Q3 = jnp.percentile(arr, 75)  # 75th percentile
    IQR = Q3 - Q1  # Interquartile range
    return IQR
get_iqr_jitted = jax.jit(get_iqr)

def get_ev_iqr(q_sum_hh, p_l, q_l, fc_l):
    ev_perct= get_ev_perct_e_jitted(q_sum_hh, p_l, q_l, fc_l)
    ev_perct_iqr = get_iqr(ev_perct)
    return ev_perct_iqr
get_ev_iqr_jitted = jax.jit(get_ev_iqr)




    
def summarize_array(arr):
    arr = arr[~np.isnan(arr) & ~np.isinf(arr)]
    if len(arr) == 0:
        return "Array has only NaN or Inf values!"
    summary = {
        'count': len(arr),
        'mean': np.nanmean(arr),
        'std': np.nanstd(arr),
        'min': np.nanmin(arr),
        '25%': np.percentile(arr, 25),
        '50% (median)': np.nanmedian(arr),
        '75%': np.percentile(arr, 75),
        'max': np.nanmax(arr),
        'variance': np.nanvar(arr)
    }
    return summary






#fc_l = fc_l0

##########################################################################################
#Achieve a goal of 20 percent of total water revenue collected from fixed minimum charges. 
##########################################################################################

def objective(param):
    param = jnp.maximum(param, 0.01)
    p_l, q_l, fc_l = param_to_pq_jitted(param)
    #p_l, q_l, lam1, lam2 = param_to_pq_jitted(param)
    jax.debug.print("Current param {y}", y= jax.device_get(param))
    #q_sum_hh = get_q_sum_hh_jitted(p_l, q_l, fc_l)
    q_sum_hh = get_q_sum_hh_jitted(p_l, q_l, fc_l)
    r = from_q_to_r_jitted(q_sum_hh, p_l, q_l, fc_l)
    result = revenue_diff_jitted(r)
    #conserve_constraint = conservation_condition_jitted(q_sum_sim, p_l, q_l, fc_l)
    #revenue_constraint = revenue_non_exceeding_jitted(r)
    #result = result - lam1*conserve_constraint - lam2*revenue_constraint 
    result_value = jax.device_get(result)
    jax.debug.print("Current Value {x}", x= result_value)
    return result
objective_jitted = jax.jit(objective)

    #q_sum_hh, q_sum_sim = get_q_sum_jitted(p_l, q_l, fc_l)
    #r = from_q_to_r_jitted(q_sum_hh, p_l, q_l, fc_l)
    #jax.debug.print("Current r[0] {x}", x= jax.device_get(r[0]))
    #gc.collect()
def conservation_constraint(param):
    #param = jnp.maximum(param, 0)
    p_l, q_l, fc_l = param_to_pq_jitted(param)
    #jax.debug.print("Current param {y}", y= jax.device_get(param))
    q_sum_sim = get_q_sum_sim_jitted(p_l, q_l, fc_l)
    #result = nansum_ignore_nan_inf_jitted(q_sum_sim - q0_sum)/len(q0_sum)
    result = conservation_condition_jitted(q_sum_sim, p_l, q_l, fc_l)
    #result_value = jax.device_get(result)
    #jax.debug.print("Current Value {x}", x= result_value)
    return result
conservation_constraint_jitted = jax.jit(conservation_constraint)

constraint1 = NonlinearConstraint(conservation_constraint_jitted, 
                                 0.95, 1.0, jac='2-point', hess=BFGS())

solution = cobyqa.minimize(objective, 
                           param0, 
                           bounds = bounds, constraints=(constraint1, constraint2, constraint3), options={'disp': True,
                                                                                                                'feasibility_tol': 0.001, 
                                                                                                        #'radius_init':0.5, 
                                                                                                        'radius_final':0.0001
                                                                                                        })

    
def revenue_non_exceeding_constraint(param):
    p_l, q_l, fc_l = param_to_pq_jitted(param)
    q_sum_hh = get_q_sum_hh_jitted(p_l, q_l, fc_l)
    r = from_q_to_r_jitted(q_sum_hh, p_l, q_l, fc_l)
    return revenue_non_exceeding_jitted(r)

revenue_non_exceeding_constraint_jitted = jax.jit(revenue_non_exceeding_constraint)

def ev_constraint(param):
    p_l, q_l, fc_l = param_to_pq_jitted(param)
    q_sum_hh = get_q_sum_hh_jitted(p_l, q_l, fc_l)
    ev_iqr = get_ev_iqr_jitted(q_sum_hh, p_l, q_l, fc_l)
    return ev_iqr

ev_constraint_jitted = jax.jit(ev_constraint)

#cons1 = {'type': 'ineq',
 #        'fun' : conservation_constraint_jitted}
#cons2 = {'type': 'ineq',
 #        'fun' : revenue_non_exceeding_constraint_jitted}
# Define the bounds as constraints
#def bound1(x):
 #   return 3.09 - x[0]
#bound1_jitted = jax.jit(bound1)

#def bound2(x):
 #   return 4 - x[5]
#bound2_jitted = jax.jit(bound2)

#def bound3(x):
 #   return 5 - x[6]
#bound3_jitted = jax.jit(bound3)

#def bound4(x):
 #   return 9 - x[7]
#bound4_jitted = jax.jit(bound4)

#cons = [
 #   {'type': 'ineq', 'fun': conservation_constraint_jitted},
  #  {'type': 'ineq', 'fun': revenue_non_exceeding_constraint_jitted},
    #{'type': 'ineq', 'fun': bound1_jitted},
    #{'type': 'ineq', 'fun': bound2_jitted},
    #{'type': 'ineq', 'fun': bound3_jitted},
    #{'type': 'ineq', 'fun': bound4_jitted}
#]

constraint1 = NonlinearConstraint(conservation_constraint_jitted, 
                                 0.95, 1.0, jac='2-point', hess=BFGS())

constraint2 = NonlinearConstraint(revenue_non_exceeding_constraint_jitted, 
                                 -1*jnp.inf, 0.0, jac='2-point', hess=BFGS())

constraint3 = NonlinearConstraint(ev_constraint_jitted, 
                                 0.0, 0.015, jac='2-point', hess=BFGS())

#cons = ([cons1])
#b_p1 = (0.1, 3.09)
#b_p = (0.1, 5)
#b_q2q1 = (0.1, 6-2)
#b_q3q2 = (0.1, 11-6)
#b_q4q3 = (0.1, 20-11)
#b = (0, jnp.inf)
#bnds = (b_p1, b_p, b_p, b_p, b_p, 
#        b_q2q1, b_q3q2, b_q4q3,
        #b, b
#        )

#bounds = Bounds([0.01, 0.01, 0.01, 0.01, 0.01, 
                 #0.01,
#                 0.01, 0.01, 0.01,
                 #0.01,
#                 10.8-8.5, 16.5-10.8, 37-16.5, 
#                 37-37
#                 ], 
#                [3.09, 10, 10, 10, 10,
                 #4,
#                 5, 9, 11,
                 #10.8,
#                 37, 37, 37, 
#                 37
#                 ])

bounds = Bounds([0.01, 0.01, 0.01, 0.01, 0.01, 
                 0.01, 
                 0.01, 0.01, 0.01,
                 0.01, 
                0.01, 0.01, 0.01,
                0.01
                ], 
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, 
                 jnp.inf, 
                 jnp.inf, jnp.inf, jnp.inf, 
                 jnp.inf, 
                 jnp.inf, jnp.inf, jnp.inf, 
                 jnp.inf
                 ])

param0 = jnp.array([3.09, 5.01-3.09, 8.54-5.01, 12.9-8.54, 14.41-12.9, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    10.8-8.5, 16.5-10.8, 37-16.5
                    , 37-37
                    #1, 1
                    ])

param_no = jnp.array([3.31, 3.93, 3.36+1, 2.29+2, 2.74+2, 
                    2.52, 
                    3.8, 5, 7.01,
                    7.45, 
                    0.01, 0.01,  18.7
                    , 0.01
                    #1, 1
                    ])

#param0 = jnp.array([5.01, 8.54-5.01, 12.9-8.54, 14.41-12.9, 3,
#                    1, 
#                    2, 4, 5,
                    #8.5, 
                    #10.8, 16.5, 37
                    #, 37
                    #1, 1
#                    ])
#param0 = jnp.array([3.09, 5.01-3.09, 8.54-5.01, 12.9-8.54, 14.41-12.9, 2, 6-2, 11-6, 20-11])
    
#solution = minimize(objective_jitted,param0, method='SLSQP',\
 #                   bounds=bnds,constraints=cons)
#array([3.09072989, 1.91835551, 3.53059349, 4.36147887, 1.50884224,
#       2.00224281, 3.99912167, 5.00152367, 8.99656216])

#conservartion_solution = minimize(conservation_constraint_jitted, param0, method = 'Nelder-Mead')

#obj_solution = minimize(objective_jitted, param0, method = 'Nelder-Mead',bounds=bnds)

#solution = minimize(objective_jitted, param0, method='trust-constr', constraints=constraints, bounds =bounds, options={'verbose': 2})
solution = cobyqa.minimize(objective, 
                           param0, 
                           bounds = bounds, constraints=(constraint1, constraint2, constraint3), options={'disp': True,
                                                                                                                'feasibility_tol': 0.001, 
                                                                                                        #'radius_init':0.5, 
                                                                                                        'radius_final':0.0001
                                                                                                        })


#solution_0 = cobyqa.minimize(objective, param0, bounds = bounds, constraints=(constraint1, constraint2), options={'disp': True})

#np.savetxt('cf_risk_result_single.csv', solution, delimiter=',')
"""