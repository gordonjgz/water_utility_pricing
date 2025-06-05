"""

Prepare and generate eta_l 


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
 """
import os
os.environ['JAX_ENABLE_X64'] = 'flase'
from jax import config
config.update("jax_enable_x64", False)

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
#eta_l = jnp.array(demand_2018_using_eta['e_diff'])

eta_l = jnp.array(demand_2018_using_eta['mean_e_diff'])

bedroom = jnp.array(demand_2018_using_new['bedroom'])
bathroom = jnp.array(demand_2018_using_new['bathroom'])
prev_NDVI = jnp.array(demand_2018_using_new['prev_NDVI'])
heavy_water_app = jnp.array(demand_2018_using_new['heavy_water_app'])

Z_current_using = jnp.column_stack((jnp.array(demand_2018_using_new['mean_TMAX_1']),
                                      jnp.array(demand_2018_using_new['IQR_TMAX_1']),
                                      jnp.array(demand_2018_using_new['total_PRCP']) 
                                      ,jnp.array(demand_2018_using_new['IQR_PRCP'])
                                      ))
Z_current = Z_current_using

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

##########################################
#### Calculate Changing NDVI with Z #####
###########################################

ndvi_df = demand_2018_using_new_season[['prem_id', 'bill_ym', 'prev_NDVI', 'mean_TMAX_1', 'IQR_TMAX_1', 'total_PRCP', 'IQR_PRCP', 'income', 'quantity']]
ndvi_df['NDVI'] = ndvi_df .groupby('prem_id')['prev_NDVI'].shift(-1)
import statsmodels.formula.api as smf

formula = 'NDVI ~ mean_TMAX_1 + IQR_TMAX_1 + total_PRCP + IQR_PRCP + income'

# Fit the OLS (Ordinary Least Squares) model
model = smf.ols(formula, data=ndvi_df)

# Get the regression results
results = model.fit()

# Print the regression summary
print("\nLinear Regression Results:")
print(results.summary())

#Linear Regression Results:
#                            OLS Regression Results                            
#==============================================================================
#Dep. Variable:                   NDVI   R-squared:                       0.127
#Model:                            OLS   Adj. R-squared:                  0.127
#Method:                 Least Squares   F-statistic:                 3.935e+04
#Date:                Mon, 05 May 2025   Prob (F-statistic):               0.00
#Time:                        11:17:28   Log-Likelihood:             1.1813e+06
#No. Observations:             1356924   AIC:                        -2.363e+06
#Df Residuals:                 1356918   BIC:                        -2.363e+06
#Df Model:                           5                                         
#Covariance Type:            nonrobust                                         
#===============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
#-------------------------------------------------------------------------------
#Intercept       0.1149      0.001     92.940      0.000       0.112       0.117
#mean_TMAX_1     0.0030   1.16e-05    256.693      0.000       0.003       0.003
#IQR_TMAX_1      0.0013    3.2e-05     40.768      0.000       0.001       0.001
#total_PRCP      0.0080   4.49e-05    177.803      0.000       0.008       0.008
#IQR_PRCP       -0.0118      0.000    -29.460      0.000      -0.013      -0.011
#income       1.449e-07   1.26e-09    114.940      0.000    1.42e-07    1.47e-07
#==============================================================================
#Omnibus:                    23337.264   Durbin-Watson:                   0.543
#Prob(Omnibus):                  0.000   Jarque-Bera (JB):            24581.074
#Skew:                          -0.324   Prob(JB):                         0.00
#Kurtosis:                       3.122   Cond. No.                     1.07e+06
#==============================================================================

#Notes:
#[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#[2] The condition number is large, 1.07e+06. This might indicate that there are
#strong multicollinearity or other numerical problems.

beta_prcp = 0.008

NDVI = demand_2018_using_new_season.groupby('prem_id')['prev_NDVI'].shift(-1)
last_dec_NDVI = demand_2018_using_new_season.groupby('prem_id')['prev_NDVI'].transform('first')
NDVI = jnp.array(NDVI.fillna(last_dec_NDVI))  ### current month NDVI with this dec filled as last dec

prev_NDVI = jnp.array(demand_2018_using_new_season['prev_NDVI'])
prem_id = jnp.array(demand_2018_using_new_season['prem_id'])
bill_ym = jnp.array(demand_2018_using_new_season['bill_ym'])

def compute_new_ndvi(Z, Z_current, NDVI, beta_prcp = beta_prcp):
    new_prcp = Z[:, 2]
    delta_prcp = new_prcp - Z_current[:, 2]
    return NDVI + beta_prcp * delta_prcp

compute_new_ndvi_jitted = jax.jit(compute_new_ndvi)

def update_prev_ndvi(new_NDVI, prem_ids, bill_ym):
    # Sort by prem_id then bill_ym
    sort_idx = jnp.lexsort((bill_ym, prem_ids))
    sorted_ndvi = new_NDVI[sort_idx]
    sorted_prem_ids = prem_ids[sort_idx]

    # Shift NDVI by 1
    shifted_ndvi = jnp.roll(sorted_ndvi, 1)

    # Identify group boundaries
    is_first = sorted_prem_ids != jnp.roll(sorted_prem_ids, 1)
    is_last = jnp.roll(sorted_prem_ids, -1) != sorted_prem_ids

    first_idxs = jnp.nonzero(is_first, size=sorted_ndvi.shape[0], fill_value=-1)[0]
    last_idxs = jnp.nonzero(is_last, size=sorted_ndvi.shape[0], fill_value=-1)[0]

    # Keep only valid indices (skip -1)
    valid = (first_idxs >= 0) & (last_idxs >= 0)
    valid_idx = jnp.nonzero(valid, size=first_idxs.shape[0], fill_value=-1)[0]
    first_idxs = first_idxs[valid_idx]
    last_idxs = last_idxs[valid_idx]

    # Create an array to overwrite first elements with last NDVI in group
    updated = shifted_ndvi.at[first_idxs].set(sorted_ndvi[last_idxs])

    # Unsort to original order
    inv_idx = jnp.argsort(sort_idx)
    prev_ndvi = updated[inv_idx]
    return prev_ndvi

update_prev_ndvi_jitted = jax.jit(update_prev_ndvi)

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
    new_ndvi = compute_new_ndvi_jitted(Z, Z_current, NDVI)
    prev_NDVI = update_prev_ndvi_jitted(new_ndvi, prem_id, bill_ym)
    
    A_o = jnp.column_stack(( 
        bathroom,
        prev_NDVI,
        ))
    
    def get_total_wk (beta_1, beta_2,
                  c_wo,
                  beta_4, 
                  c_a,
                  beta_6,
                  c_r,
                  k, 
                  Z,
                  #A_i = A_current_income,
                 # A_p = A_current_price,
                  #A_o = A_current_outdoor,
                  #G = G,
                  p = p_l, I = I,
                  p0 =p0, 
                  de = de,
                  ):
        
        A_i = jnp.column_stack((
            heavy_water_app,
            bedroom, 
            prev_NDVI, 
        ))
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
    #gc.collect()
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

hh_size = len(np.unique(np.array(demand_2018_using_new_season['prem_id'], dtype = np.int32)))
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
        prem_id =  jnp.array(demand_2018_using_new_season['prem_id'], dtype = jnp.int32)
    
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

mean_nu_array =jnp.mean(nu_array, axis = 1)

demand_eta = demand_2018_using_new_season[['prem_id', 'bill_ym']]

demand_eta['eta'] = eta_l

demand_eta['mean_eta'] = demand_eta.groupby('prem_id')['eta'].transform('mean')

mean_eta_l = jnp.array(demand_eta['mean_eta'])

def get_expenditure_in_v_out(q_sum_hh, p_l, q_l, fc_l, Z):
    new_ndvi = compute_new_ndvi_jitted(Z, Z_current, NDVI)
    prev_NDVI = update_prev_ndvi_jitted(new_ndvi, prem_id, bill_ym)
    
    A_o = jnp.column_stack(( 
        bathroom,
        prev_NDVI,
        ))
    
    A_p= jnp.column_stack((
        bedroom, 
        prev_NDVI, 
        Z[:, 0],
        Z[:, 2],
    ))
    A_i = jnp.column_stack((
        heavy_water_app,
        bedroom, 
        prev_NDVI, 
    ))
    alpha = jnp.exp(jnp.dot(A_p, b4)
                + c_alpha
            )
    rho = abs(jnp.dot(A_i, b6)
                + c_rho
                )
    p = get_current_marginal_p_jitted(q_sum_hh, p_l, q_l, fc_l)
    exp_factor = jnp.exp(jnp.dot(A_o, b1) + jnp.dot(Z, b2) + c_o + mean_eta_l+mean_nu_array)
    tolerance = 1e-3 # Define a small tolerance for numerical stability
    alpha_minus_1 = alpha - 1.0
    # Calculate the term related to price, handling alpha close to 1
    # The term needed for 'result' is exp_factor * [pk^(1-alpha)/(1-alpha)]
    # When alpha is close to 1, this should be exp_factor * log(pk) (as per the limit of the first term of V)
    price_component = jnp.where(
        jnp.abs(alpha_minus_1) < tolerance,
        jnp.log(p), # Limit case for [pk^(1-alpha)/(1-alpha)] when alpha is close to 1
        jnp.divide(jnp.power(p, -alpha_minus_1), -alpha_minus_1) # Original calculation
    )
    result = exp_factor * price_component
    return result, rho
get_expenditure_in_v_out_jitted = jax.jit(get_expenditure_in_v_out)

def get_v_out(q_sum_hh, p_l, q_l, fc_l, Z):
    exp_v, rho = get_expenditure_in_v_out_jitted(q_sum_hh, p_l, q_l, fc_l, Z)
    sim_result_Ik = get_virtual_income_jitted(q_sum_hh, p_l, q_l, fc_l)

    tolerance = 1e-3 # Use the same tolerance
    rho_minus_1 = rho - 1.0
    
    # Calculate the second term of V, handling rho close to 1
    # The term is (I+d_k)^(1-rho) / (1-rho)
    second_term_V = jnp.where(
        jnp.abs(rho_minus_1) < tolerance,
        jnp.log(sim_result_Ik), # Limit case when rho is close to 1
        jnp.divide(jnp.power(sim_result_Ik, -rho_minus_1), -rho_minus_1) # Original calculation
    )
    
    # Total V = -exp_v + second_term_V
    # Note: exp_v is the first term of V *without* the leading negative sign
    v_out = -1 * exp_v + second_term_V
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
#Array(7185366.2319867, dtype=float64)
r_agg_0 = nansum_ignore_nan_inf_jitted(r0 )/12
#Array(28058192.92994599, dtype=float64)

#q_sum_hh_1417 = get_q_sum_hh_jitted(p_l0, q_l0, fc_l0, Z_1417)
#q0_filtered = q_sum_hh_current[q_sum_hh_current < 150000]
#q0_filtered = q_sum_hh_history[q_sum_hh_history < 150000]
q_agg_history = nansum_ignore_nan_inf_jitted(q_sum_hhhistory/100)/12
# Array(721182.51175329, dtype=float64)
q_agg_0 = nansum_ignore_nan_inf_jitted(q_sum_hh0/100)/12
#Array(2276729.1038763 dtype=float64)

cs_history = get_v_out_jitted(q_sum_hhhistory , p_l0, q_l0, fc_l0, Z_1417)
#cs0_filtered = cs_0[(cs_0 > -0.5*1e9) ]
cs_agg_history = nansum_ignore_nan_inf_jitted(cs_history)/12
#Array(1.35726374e+09, dtype=float64)
cs_0 = get_v_out_jitted(q_sum_hh0, p_l0, q_l0, fc_l0, Z_current)
cs_agg_0= nansum_ignore_nan_inf_jitted(cs_0)/12
#Array(1.33489888e+09, dtype=float64)

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

######################
#### EV and CV #####
########################

mp_0 = jnp.where(CAP == 1,p_l0_CAP[k0], p_l0[k0])

def find_q_sum_hh_close_to_ql(all_q_sum_hh, q_l, tolerance=1e-3):
    """
    Finds indices of scaled q_sum_hh values close to any element in q_l
    and returns the nearest q_l element and its index for each value.

    Args:
        all_q_sum_hh: JAX array of q_sum_hh values to check.
        q_l: JAX array of q_l elements (e.g., 4 elements).
        sim: Scalar or array to divide all_q_sum_hh by.
        tolerance: The tolerance for considering values "close".

    Returns:
        A tuple containing:
        - is_close_to_any_ql: Boolean JAX array of shape (N,), where True indicates the scaled
          value is close to an element in q_l.
        - nearest_ql_indices: Integer JAX array of shape (N,), containing the index
          in q_l that was nearest to the corresponding scaled q_sum_hh value.
        - nearest_ql_values: JAX array of shape (N,), containing the value from
          q_l that was nearest to the corresponding scaled q_sum_hh value.
    """
    # Scale the input q_sum_hh values
    q_hh = all_q_sum_hh / sim

    # Reshape for broadcasting: [num_q_sum_hh, 1] vs [1, num_ql]
    q_hh_reshaped = q_hh[:, None]
    q_l_reshaped = q_l[None, :]

    # Calculate absolute difference between each q_hh and each q_l element
    diffs = jnp.abs(q_hh_reshaped - q_l_reshaped)

    # Check if the difference is within tolerance for any q_l element
    is_close_to_any_ql = jnp.any(diffs < tolerance, axis=1)

    # Find the index of the minimum difference for each q_hh value
    # This gives the index in q_l that is nearest to each q_hh
    nearest_ql_indices = jnp.argmin(diffs, axis=1)

    # Get the actual nearest q_l values using the found indices
    nearest_ql_values = q_l[nearest_ql_indices]

    return is_close_to_any_ql, nearest_ql_indices, nearest_ql_values

find_q_sum_hh_close_to_ql_jitted = jax.jit(find_q_sum_hh_close_to_ql)

@jax.jit
def solve_for_pbar_vectorized(tk, A, alpha, rho, pk, vi0):
    #tolerance = 1e-6  # threshold for "closeness" to 1

    def scalar_solver(tk_, A_, alpha_, rho_, pk_,vi0_, eps=1e-6):
        log_tk = jnp.log(tk_)
        def f(pbar):
            # Use epsilon cutoff to trigger limit cases
            near_alpha_1 = jnp.abs(1.0 - alpha_) < eps
            near_rho_1 = jnp.abs(1.0 - rho_) < eps

            # Power difference for alpha ≠ 1
            p_diff = jnp.where(
                near_alpha_1,
                jnp.log(pbar) - jnp.log(pk_),
                pbar**(1 - alpha_) - pk_**(1 - alpha_)
            )

            # Multiplier for inner log term
            alpha_term = jnp.where(
                near_alpha_1,
                1.0,  # lim_{α→1} (1 - α)/(1 - ρ) * (log p̄ - log pk)
                (1 - alpha_) / (1 - rho_)
            )

            # Additive term (I + d0)^{1 - rho}
            Id_term = jnp.where(
                near_rho_1,
                jnp.log(vi0_),  # lim_{ρ→1} log(I + d0)
                (vi0_)**(1 - rho_)
            )

            # Inner log argument
            inner = jnp.where(
                near_rho_1,
                jnp.exp(A_) * alpha_term * p_diff + Id_term,
                jnp.exp(A_) * alpha_term * p_diff + Id_term
            )
            inner = jnp.maximum(inner, 1e-12)  # prevent log(0)

            # Full function expression
            f_val = A_ - alpha_ * jnp.log(pbar)
            f_val += jnp.where(
                near_rho_1,
                rho_ * jnp.log(inner),  # lim_{ρ→1}
                (rho_ / (1 - rho_)) * jnp.log(inner)
            )
            return f_val - log_tk

        # Initial bounds
        lower = pk_ * 0.5
        upper = pk_ * 1.5

        def cond(val):
            l, u, _, i = val
            return (u - l > 1e-6) & (i < 100)

        def body(val):
            l, u, _, i = val
            m = (l + u) / 2
            f_m = f(m)
            f_l = f(l)
            same_sign = f_l * f_m >= 0
            new_l = jnp.where(same_sign, m, l)
            new_u = jnp.where(same_sign, u, m)
            return (new_l, new_u, m, i + 1)

        _, _, root, _ = jax.lax.while_loop(cond, body, (lower, upper, 0.0, 0))
        return root

    return jax.vmap(scalar_solver)(tk, A, alpha, rho, pk, vi0)

vi_0 = get_virtual_income_jitted(q_sum_hh0, p_l0, q_l0, fc_l0)

def get_e_new_v_p0(q_sum_hh, p_l, q_l, fc_l, Z):
    v_out = get_v_out_jitted(q_sum_hh, p_l, q_l, fc_l, Z)
    new_ndvi = compute_new_ndvi_jitted(Z, Z_current, NDVI)
    prev_NDVI = update_prev_ndvi_jitted(new_ndvi, prem_id, bill_ym)
    is_close_to_any_ql, nearest_ql_indices, nearest_ql_values = find_q_sum_hh_close_to_ql_jitted(q_sum_hh, q_l)
    
    A_o = jnp.column_stack(( 
        bathroom,
        prev_NDVI,
        ))
    
    A_p= jnp.column_stack((
        bedroom, 
        prev_NDVI, 
        Z[:, 0],
        Z[:, 2],
    ))
    A_i = jnp.column_stack((
        heavy_water_app,
        bedroom, 
        prev_NDVI, 
    ))
    alpha = jnp.exp(jnp.dot(A_p, b4)
                + c_alpha
            )
    rho = abs(jnp.dot(A_i, b6)
                + c_rho
                )
    A_term = jnp.exp(jnp.dot(A_o, b1) + jnp.dot(Z, b2) + c_o+eta_l)
    tolerance = 1e-3 # Use the same tolerance
    alpha_minus_1 = alpha - 1.0


    pk = get_current_marginal_p(q_sum_hh, p_l, q_l, fc_l)
    ###########################
    # Efficient p_bar solving #
    ###########################

    # Default: mp = pk
    mp = mp_0

    tk_safe     = jnp.where(is_close_to_any_ql, nearest_ql_values, 1.0)  # dummy tk
    A_safe      = jnp.where(is_close_to_any_ql, A_term, 0.0)
    alpha_safe  = jnp.where(is_close_to_any_ql, alpha, 1.5)              # Avoid alpha=1 edge case
    rho_safe    = jnp.where(is_close_to_any_ql, rho, 0.5)
    pk_safe     = jnp.where(is_close_to_any_ql, pk, 1.0)
    vi0_safe     = jnp.where(is_close_to_any_ql, vi_0, 1.0)     

    # Solve only at kink rows
    p_bar_solved = solve_for_pbar_vectorized(tk_safe, A_safe, alpha_safe, rho_safe, pk_safe, vi0_safe)

    # Blend with default using mask
    mp = jnp.where(is_close_to_any_ql, p_bar_solved, mp_0)
    
    mp_term = jnp.where(
        jnp.abs(alpha_minus_1) < tolerance,
        jnp.log(mp), # Limit case when rho is close to 1
        jnp.divide(jnp.power(mp, -alpha_minus_1), -alpha_minus_1) # Original calculation
    )
    
    # Calculate the base for the final power calculation
    # The term inside the power is (1-rho) * (v_out + A_term * mp_term)
    base_arg = jnp.multiply((1 - rho), (v_out + jnp.multiply(A_term, mp_term)))

    # Ensure the base is non-negative and not too small to avoid issues with jnp.power
    base = jnp.maximum(base_arg, 1e-16)

    # Handle the case where rho is close to 1 separately for numerical stability.
    # As rho approaches 1, the term ((1-rho) * Y)^(1/(1-rho)) approaches 0.
    # We use jnp.where to return 0 directly when 1-rho is close to zero.
    e_out_unscaled = jnp.where(
        jnp.abs(1 - rho) < tolerance,
        0.0, # Return 0 when rho is close to 1
        jnp.power(base, jnp.divide(1.0, (1 - rho))) # Original power calculation otherwise
    )
    
    additional = jnp.where(is_close_to_any_ql, jnp.multiply((p_bar_solved - mp_0), q_l0[k0]) ,0.0)

    # Apply the scaling factor
    e_out = jnp.multiply(e_out_unscaled-additional, de)
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
    
def get_ev(q_sum_hh, p_l, q_l, fc_l,Z):
    e = get_e_new_v_p0_jitted(q_sum_hh, p_l, q_l, fc_l, Z)
    #diff_payment = get_diff_payment_jitted(q_sum_hh, p_l, q_l, fc_l)
    ev = e - vi_0
    return ev
get_ev_jitted = jax.jit(get_ev)

imu_0 = jnp.power(jnp.maximum(jnp.multiply(d_k0, de) + I, 1e-16), rho)

def get_compensating_variation (cs1):
    """
    This arises from a first-order Taylor approximation of the expenditure function, and assumes that:
    Utility is quasi-linear or homothetic in income (which the utility form approximates),
    The marginal utility of income doesn't change too drastically over the utility change — i.e., no big income effect "kinks".
    """
    diff_cs = cs1 - cs_0
    mu_income = jnp.power(jnp.maximum(jnp.multiply(d_k0, de) + I, 1e-16), rho)
    result = jnp.multiply(diff_cs, mu_income)
    return result
get_compensating_variation_jitted = jax.jit(get_compensating_variation)

def cara_jitted(x, a):
  """
  Calculates the CARA utility function.
  Args:
    x: Revenue/cost value (scalar or JAX array). Must be > 0.
    a: Absolute risk aversion parameter (scalar, a > 0).
  Returns:
    The CARA utility value(s) (will be negative).
  """
  # For numerical stability and economic relevance, x should be positive.
  # CARA utility is negative and increasing.
  return -jnp.exp(-a * x)

def inverse_cara_jitted(y, a):
  """
  Calculates the inverse of the CARA utility function.
  Args:
    y: The CARA utility value(s) (scalar or JAX array). Must be < 0.
    a: Absolute risk aversion parameter (scalar, a > 0).
  Returns:
    The original revenue/cost value(s).
  """
  # For valid inverse, y must be negative (output of cara_jitted).
  # jnp.log requires positive argument, so we use -y.
  return -1.0/a * jnp.log(-y)

########################
#### Revenue Conditions #####
########################
rhistory_sum_filtered =  nansum_ignore_nan_inf_jitted(rhistory)
r0_sum_filtered =  nansum_ignore_nan_inf_jitted(r0 )

rhistory_soft_constraint = 0.8*rhistory_sum_filtered

r0_soft_constraint = 0.8*r0_sum_filtered

log_r0_mean_filtered = jnp.log(r0_sum_filtered/12)

gamma = 0.3

a_val = 1e-6

r0_mean_filtered = r0_sum_filtered/len_transactions

crra_r0_mean_filtered = crra_jitted(r0_sum_filtered/len_transactions, gamma)

cara_r0_mean_filtered = cara_jitted(r0_sum_filtered/len_transactions, a_val)
    
def revenue_compare(r, r0_benchmark):
    ### Here r is for each transaction, compare to r0 as the avg cost for each transaction
    return r - r0_benchmark

revenue_compare_jitted = jax.jit(revenue_compare)

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

def cf_w_ci_crra(q_sum_sim_month, p_l, q_l, fc_l, q0_benchmark =q0_sum_mean):
    q0_benchmark_month = q0_benchmark/12
    q0_benchmark_month_crra = crra_jitted(q0_benchmark_month, gamma)
    q_sum_sim_month_crra = crra_jitted(q_sum_sim_month, gamma)
    q_sum_sim_month_crra_mean=jnp.mean(q_sum_sim_month_crra, axis=0)
    
    condition= (q0_benchmark_month_crra - q_sum_sim_month_crra_mean)>0
    return jnp.count_nonzero(condition)
cf_w_ci_crra_jitted = jax.jit(cf_w_ci_crra)

def conservation_condition(q_sum_sim, p_l, q_l, fc_l):
    num_satisfying = cf_w_ci_jitted(q_sum_sim, p_l, q_l, fc_l)
    #total = q_sum_sim.shape[0] * q_sum_sim.shape[1]  # 12 * 100 = 1200
    #return num_satisfying / total
    return num_satisfying/(sim)
conservation_condition_jitted = jax.jit(conservation_condition)

def conservation_condition_crra(q_sum_sim_month, p_l, q_l, fc_l):
    num_satisfying = cf_w_ci_crra_jitted(q_sum_sim_month, p_l, q_l, fc_l)
    return num_satisfying/(sim)
conservation_condition_crra_jitted = jax.jit(conservation_condition_crra)

######################
#### Optimization #####
########################

### Use a linear loss function to control true risk neutral vs risk averse. 
### quadratic loss function already adding risk averse. 


def loss_function(x):
    return jnp.maximum(0.0, -1*x)

loss_function_jitted = jax.jit(loss_function)

def loss_function_quadratic(x):
  """
  Calculates a quadratic penalty for negative values of x (constraint violation).

  Args:
    x: The margin (e.g., revenue - cost). Penalty applies if x < 0.

  Returns:
    0 if x is non-negative (constraint met).
    The square of the absolute value of x if x is negative (constraint violated).
  """
  # Calculate the positive magnitude of the shortfall
  shortfall_magnitude = jnp.maximum(0.0, -1*x)

  # Square the shortfall magnitude to make the penalty quadratic
  return shortfall_magnitude**2

loss_function_quadratic_jitted = jax.jit(loss_function_quadratic)

lambda_ = 0.5

#r0_benchmark =r0_sum_filtered/12
#r0_benchmark_crra =crra_jitted(r0_sum_filtered/12, gamma)

def inverse_crra(y, gamma):
  """
  Calculates the inverse of the CRRA utility function.

  Args:
    y: The value(s) from the CRRA utility space (output of CRRA function).
       Can be a scalar or a JAX array.
    gamma: The CRRA risk aversion parameter (scalar). Must be >= 0.

  Returns:
    The corresponding value(s) in the original units (e.g., revenue, cost).
    Will have the same shape as y.
  """
  # Ensure gamma is a JAX array for conditional logic
  gamma_arr = jnp.asarray(gamma)

  # Handle the gamma = 1 case (log utility)
  # Use jnp.isclose for floating point comparison
  is_log_case = jnp.isclose(gamma_arr, 1.0)

  # Calculate inverse for gamma != 1
  # Use jnp.where to handle element-wise selection based on condition
  # Ensure the term inside the power is non-negative if needed, although
  # the range of CRRA for x > 0 typically ensures this for standard gamma.
  # Adding a small epsilon might be necessary for numerical stability if y*(1-gamma) can be near zero.
  # However, for typical CRRA usage with positive x, y*(1-gamma) should be positive
  # if gamma < 1, and negative if gamma > 1. The power (1/(1-gamma)) handles this.
  # Let's stick to the direct formula first.
  inverse_non_log = (y * (1.0 - gamma_arr))**(1.0 / (1.0 - gamma_arr))

  # Calculate inverse for gamma = 1 (exponential)
  inverse_log = jnp.exp(y)

  # Combine results based on the condition
  # jnp.where(condition, value_if_true, value_if_false)
  x = jnp.where(is_log_case, inverse_log, inverse_non_log)

  return x

inverse_crra_jitted = jax.jit(inverse_crra)

def get_result (p_l, q_l, fc_l, Z):
    q_sum_hh = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z)
    r = from_q_to_r_jitted(q_sum_hh, p_l, q_l, fc_l)
    r_sum_filtered = nansum_ignore_nan_inf_jitted(r)
    ev = get_ev_jitted(q_sum_hh, p_l, q_l, fc_l, Z)
    ev_ratio_fullyear = nansum_ignore_nan_inf_jitted(jnp.divide(ev, I))
    loss = -1 * lambda_* loss_function_jitted(r_sum_filtered - r0_sum_filtered)
    result =ev_ratio_fullyear + loss
    return result

get_result_jitted = jax.jit(get_result)

def get_result_crra (p_l, q_l, fc_l, Z):
    q_sum_hh = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z)
    r = from_q_to_r_jitted(q_sum_hh, p_l, q_l, fc_l)
    # Apply mask after slicing
    monthly_sum_r = jnp.array([
        (r[padded_month_indices[i]] * mask[i]).sum()
        for i in range(12)
    ])   
    crra_r_monthly = crra_jitted(monthly_sum_r, gamma) # Array of f(R_m) (shape num_months,)
    crra_r_monthly_avg = jnp.mean(crra_r_monthly) # Average of f(R_m) (scalar)
    ce_avg_monthly_revenue = inverse_crra_jitted(crra_r_monthly_avg, gamma) # CE(bar{R}_month) (dollars/month)
    ev = get_ev_jitted(q_sum_hh, p_l, q_l, fc_l, Z)
    ev_ratio_fullyear = nansum_ignore_nan_inf_jitted(jnp.divide(ev, I))
    loss = -1* lambda_*loss_function_jitted((12.0 * ce_avg_monthly_revenue) - r0_sum_filtered) # max(0, C_total_val - 12 * CE(bar{R}_month)) (dollars)
    result =ev_ratio_fullyear + loss
    return result

get_result_crra_jitted = jax.jit(get_result_crra)


def get_result_quadratic (p_l, q_l, fc_l, Z):
    q_sum_hh = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z)
    r = from_q_to_r_jitted(q_sum_hh, p_l, q_l, fc_l)
    monthly_sum_r = jnp.array([
        (r[padded_month_indices[i]] * mask[i]).sum()
        for i in range(12)
    ])
    loss = -1 * lambda_*jnp.sum(loss_function_quadratic_jitted(monthly_sum_r - r0_sum_filtered/12))
    ev = get_ev_jitted(q_sum_hh, p_l, q_l, fc_l, Z)
    #cs = get_v_out(q_sum_hh, p_l, q_l, fc_l, Z)
    #cv = get_compensating_variation_jitted(cs)
    ev_ratio_fullyear = nansum_ignore_nan_inf_jitted(jnp.divide(ev, I))
    result =ev_ratio_fullyear + loss
    return result

get_result_quadratic_jitted = jax.jit(get_result_quadratic)

def get_result_cara (p_l, q_l, fc_l, Z):
    q_sum_hh = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z)
    r = from_q_to_r_jitted(q_sum_hh, p_l, q_l, fc_l)
    # Apply mask after slicing
    monthly_sum_r = jnp.array([
        (r[padded_month_indices[i]] * mask[i]).sum()
        for i in range(12)
    ])
    # Apply CARA function element-wise to monthly total revenues
    cara_r_monthly = cara_jitted(monthly_sum_r, a_val) # Array of f(R_m) (shape num_months,)
    # Calculate Average of f(R_m) over the 12 months
    cara_r_monthly_avg = jnp.mean(cara_r_monthly) # Average of f(R_m) (scalar, negative)
    # Calculate Empirical Certainty Equivalent of Average Monthly Revenue (dollars/month)
    # CE(bar{R}_month) = f^{-1}(Average of f(R_m))
    ce_avg_monthly_revenue = inverse_cara_jitted(cara_r_monthly_avg, a_val) # CE(bar{R}_month) (dollars/month)
    ev = get_ev_jitted(q_sum_hh, p_l, q_l, fc_l, Z)
    ev_ratio_fullyear = nansum_ignore_nan_inf_jitted(jnp.divide(ev, I))
    ce_monthly_shortfall_penalty_amount = loss_function_jitted((12.0 * ce_avg_monthly_revenue) - r0_sum_filtered) # max(0, C_total_val - 12 * CE(bar{R}_month)) (dollars)
    # Total penalty term (dollars)
    loss = -1* lambda_ * ce_monthly_shortfall_penalty_amount
    result =ev_ratio_fullyear + loss
    return result

get_result_cara_jitted = jax.jit(get_result_cara)

###############################################################
######### Generate Monte Carlo Result #####################
###############################################################
log_sd_perturb=1

######### Concave for intra-year #####################

# The modified get_result_crra function, incorporating Monte Carlo
@jax.jit # JIT the entire get_result_crra function
def get_result_crra_monte_carlo(p_l, q_l, fc_l, Z_original,
                                total_num_samples=20, log_std_dev_factor=log_sd_perturb,
                                min_precip_epsilon=1e-6, rng_key=jax.random.PRNGKey(41),
                                column_index=2, # Removed batch_size
                                # Pass external CRRA specific parameters here:
                                padded_month_indices_param=padded_month_indices,
                                mask_param = mask, gamma_param = gamma):
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
        print("Warning: No rng_key provided. Using a default JAX PRNGKey(0).")

    # 1. Prepare common parameters for noise generation and reconstruction
    column_to_perturb_base = Z_original[:, column_index]
    column_positive_base = jnp.where(column_to_perturb_base <= 0, min_precip_epsilon, column_to_perturb_base)
    log_column_base = jnp.log(column_positive_base)
    sigma_log = log_std_dev_factor

    # Store p_l, q_l, fc_l in a tuple/array to pass around easily
    param_for_core = (p_l, q_l, fc_l)

    # Generate all keys needed for total_num_samples
    all_keys = jax.random.split(rng_key, total_num_samples)

    # Define a helper function to process a single Monte Carlo sample given its unique key
    def process_single_mc_sample(key):
        # Generate normal noise for this single sample
        normal_noise_single = jax.random.normal(key, shape=log_column_base.shape) * sigma_log
        log_perturbed_column_single = log_column_base + normal_noise_single
        perturbed_column_single = jnp.exp(log_perturbed_column_single)

        # Reconstruct the full Z array for this single sample
        Z_sample_full_array = Z_original.at[:, column_index].set(perturbed_column_single)

        # Run the core calculation for this sample, passing the extra args
        return _single_mc_run_get_result_crra_core(
            param_for_core, Z_sample_full_array,
            padded_month_indices_param, mask_param, gamma_param
        )

    # Use jax.vmap directly to process all samples in parallel
    monte_carlo_results_per_sample = jax.vmap(process_single_mc_sample)(all_keys)

    # Take the average of all Monte Carlo results
    average_mc_result = jnp.mean(monte_carlo_results_per_sample)
    del monte_carlo_results_per_sample
    return average_mc_result

# The modified get_result function, incorporating Monte Carlo
@jax.jit # JIT the entire get_result function
def get_result_monte_carlo(p_l, q_l, fc_l, Z_original,
                           total_num_samples=20, log_std_dev_factor=log_sd_perturb,
                           min_precip_epsilon=1e-6, rng_key=jax.random.PRNGKey(41),
                           column_index=2): # Removed batch_size as it's not needed for vmap
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
        print("Warning: No rng_key provided. Using a default JAX PRNGKey(0).")

    # 1. Prepare common parameters for noise generation and reconstruction
    column_to_perturb_base = Z_original[:, column_index]
    column_positive_base = jnp.where(column_to_perturb_base <= 0, min_precip_epsilon, column_to_perturb_base)
    log_column_base = jnp.log(column_positive_base)
    sigma_log = log_std_dev_factor

    # Store p_l, q_l, fc_l in a tuple/array to pass around easily
    param_for_core = (p_l, q_l, fc_l)

    # Generate all keys needed for total_num_samples
    all_keys = jax.random.split(rng_key, total_num_samples)

    # Define a helper function to process a single Monte Carlo sample given its unique key
    def process_single_mc_sample(key):
        # Generate normal noise for this single sample
        normal_noise_single = jax.random.normal(key, shape=log_column_base.shape) * sigma_log
        log_perturbed_column_single = log_column_base + normal_noise_single
        perturbed_column_single = jnp.exp(log_perturbed_column_single)

        # Reconstruct the full Z array for this single sample
        Z_sample_full_array = Z_original.at[:, column_index].set(perturbed_column_single)

        # Run the core calculation for this sample
        return _single_mc_run_get_result_core(param_for_core, Z_sample_full_array)

    # Use jax.vmap directly to process all samples in parallel
    # This will create an intermediate array of all results.
    monte_carlo_results_per_sample = jax.vmap(process_single_mc_sample)(all_keys)

    # Take the average of all Monte Carlo results
    average_mc_result = jnp.mean(monte_carlo_results_per_sample)
    del monte_carlo_results_per_sample
    return average_mc_result

# The single Monte Carlo run core logic for get_result_crra
def _single_mc_run_get_result_crra_core(param_pl_ql_fcl, Z_single_sample_full_array,
                                        padded_month_indices_arg, mask_arg, gamma_arg):
    p_l, q_l, fc_l = param_pl_ql_fcl

    q_sum_hh = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_single_sample_full_array)
    r = from_q_to_r_jitted(q_sum_hh, p_l, q_l, fc_l)

    monthly_sum_r_list = []
    for i in range(12):
        sliced_r = r[padded_month_indices_arg[i]]
        masked_sliced_r = sliced_r * mask_arg[i] # mask should be float or bool for mult
        monthly_sum_r_list.append(masked_sliced_r.sum())
    monthly_sum_r = jnp.array(monthly_sum_r_list)

    crra_r_monthly = crra_jitted(monthly_sum_r, gamma_arg)
    crra_r_monthly_avg = jnp.mean(crra_r_monthly)
    ce_avg_monthly_revenue = inverse_crra_jitted(crra_r_monthly_avg, gamma_arg)

    ev = get_ev_jitted(q_sum_hh, p_l, q_l, fc_l, Z_single_sample_full_array)
    ev_ratio_fullyear = nansum_ignore_nan_inf_jitted(jnp.divide(ev, I))

    loss = -1 * lambda_ * loss_function_jitted((12.0 * ce_avg_monthly_revenue) - r0_sum_filtered)
    result = ev_ratio_fullyear + loss
    return result

######### Concave for all simulation #####################

# The single Monte Carlo run logic
def _single_mc_run_get_result_core(param_pl_ql_fcl, Z_single_sample_full_array):
    """
    This contains the original core logic of get_result, but operates on a single
    perturbed Z sample. It returns the 'result'.
    """
    p_l, q_l, fc_l = param_pl_ql_fcl # Unpack parameters for clarity

    q_sum_hh = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_single_sample_full_array)
    r = from_q_to_r_jitted(q_sum_hh, p_l, q_l, fc_l)
    r_sum_filtered = nansum_ignore_nan_inf_jitted(r)

    ev = get_ev_jitted(q_sum_hh, p_l, q_l, fc_l, Z_single_sample_full_array)
    ev_ratio_fullyear = nansum_ignore_nan_inf_jitted(jnp.divide(ev, I))

    loss = -1 * lambda_ * loss_function_jitted(r_sum_filtered - r0_sum_filtered)
    result = ev_ratio_fullyear + loss
    return result


# The single Monte Carlo run core logic for get_result_crra
def _single_mc_run_get_result_crra_combined_core(param_pl_ql_fcl, Z_single_sample_full_array,
                                                  padded_month_indices_arg, mask_arg, gamma_arg):
    p_l, q_l, fc_l = param_pl_ql_fcl
    q_sum_hh = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_single_sample_full_array)

    ev = get_ev_jitted(q_sum_hh, p_l, q_l, fc_l, Z_single_sample_full_array)
    ev_ratio_fullyear = nansum_ignore_nan_inf_jitted(jnp.divide(ev, I))
    
    r = from_q_to_r_jitted(q_sum_hh, p_l, q_l, fc_l)

    monthly_sum_r_list = []
    for i in range(12):
        sliced_r = r[padded_month_indices_arg[i]]
        masked_sliced_r = sliced_r * mask_arg[i] # mask should be float or bool for mult
        monthly_sum_r_list.append(masked_sliced_r.sum())
    monthly_sum_r = jnp.array(monthly_sum_r_list)
    
    crra_r_monthly = crra_jitted(monthly_sum_r, gamma_arg)

    result = ev_ratio_fullyear, crra_r_monthly
    return result



# NEW FUNCTION: Step 1 - Generate all perturbed Z samples
@jax.jit
def generate_mc_perturbed_Z_samples(
    Z_original_base: jax.Array,
    total_num_samples=25,
    log_std_dev_factor = log_sd_perturb,
    min_precip_epsilon = jnp.array(1e-6, dtype=jnp.float32),
    rng_key= jax.random.PRNGKey(42),
    column_index = 2
) -> jax.Array:
    """
    Generates a batch of perturbed Z arrays for Monte Carlo simulation.
    This function should be called once when Z_original_base changes.
    """
    # 1. Prepare common parameters for noise generation and reconstruction
    column_to_perturb_base = Z_original_base[:, column_index]
    column_positive_base = jnp.where(column_to_perturb_base <= 0, min_precip_epsilon, column_to_perturb_base)
    log_column_base = jnp.log(column_positive_base)
    sigma_log = log_std_dev_factor

    # Generate all keys needed for total_num_samples
    all_keys = jax.random.split(rng_key, total_num_samples)

    # Define a helper function to generate a single perturbed Z sample
    def _generate_single_perturbed_Z(key):
        normal_noise_single = jax.random.normal(key, shape=log_column_base.shape) * sigma_log
        log_perturbed_column_single = log_column_base + normal_noise_single
        perturbed_column_single = jnp.exp(log_perturbed_column_single)
        # Reconstruct the full Z array for this single sample
        return Z_original_base.at[:, column_index].set(perturbed_column_single)

    # Use jax.vmap directly to generate all samples in parallel
    # This will return a (total_num_samples, num_rows, num_cols) array
    perturbed_Z_samples_batch = jax.vmap(_generate_single_perturbed_Z)(all_keys)
    return perturbed_Z_samples_batch

# MODIFIED FUNCTION: Step 2 - Calculate the Monte Carlo result from pre-generated samples
@jax.jit
def get_mc_result_from_perturbed_Z(
    p_l: jax.Array,
    q_l: jax.Array,
    fc_l: jax.Array,
    Z_perturbed_samples_batch: jax.Array # Now takes the pre-generated batch
) -> jax.Array:
    """
    Calculates the average Monte Carlo result from a pre-generated batch of perturbed Z arrays.
    This function can be called repeatedly with the same Z_perturbed_samples_batch
    for different p_l, q_l, fc_l in an optimization loop.
    """
    # Store p_l, q_l, fc_l in a tuple/array to pass around easily
    param_for_core = (p_l, q_l, fc_l)

    # Use jax.vmap directly to process all samples in parallel
    # _single_mc_run_get_result_core is mapped over the first dimension (total_num_samples)
    monte_carlo_results_per_sample = jax.vmap(_single_mc_run_get_result_core, in_axes=(None, 0))(
        param_for_core,
        Z_perturbed_samples_batch
    )
    # Take the average of all Monte Carlo results
    average_mc_result = jnp.mean(monte_carlo_results_per_sample)
    return average_mc_result

@jax.jit # JIT the entire get_result_crra function
def get_mc_result_crra_from_perturbed_Z(p_l: jax.Array,
    q_l: jax.Array,
    fc_l: jax.Array,
    Z_perturbed_samples_batch: jax.Array, # Now takes the pre-generated batch
    padded_month_indices_param=padded_month_indices,
    mask_param = mask, gamma_param = gamma):

    # Store p_l, q_l, fc_l in a tuple/array to pass around easily
    param_for_core = (p_l, q_l, fc_l)
    #monte_carlo_results_per_sample = jax.vmap(_single_mc_run_get_result_crra_core, in_axes=(None, 0))(
    #    param_for_core, Z_perturbed_samples_batch,
    #    padded_month_indices_param, mask_param, gamma_param
    #)
    
    monte_carlo_results_per_sample_ev, monte_carlo_results_per_sample_rm = jax.vmap(_single_mc_run_get_result_crra_combined_core, in_axes=(None, 0, None, None, None))(
        param_for_core, Z_perturbed_samples_batch,
        padded_month_indices_param, mask_param, gamma_param
    )
    
    crra_r_monthly_avg = jnp.mean(monte_carlo_results_per_sample_rm)
    ce_avg_monthly_revenue = inverse_crra_jitted(crra_r_monthly_avg, gamma_param)

    loss = -1 * lambda_ * loss_function_jitted((12.0 * ce_avg_monthly_revenue) - r0_sum_filtered)
    
    # Take the average of all Monte Carlo results
    average_mc_result = jnp.mean(monte_carlo_results_per_sample_ev) + loss
    del monte_carlo_results_per_sample_ev, monte_carlo_results_per_sample_rm
    return average_mc_result



param0 = jnp.array([3.09, 5.01-3.09, 8.54-5.01, 12.9-8.54, 14.41-12.9, 
                   # 2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    10.8-8.5, 16.5-10.8, 37-16.5
                    , 37-37
                    ])


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


def objective0(param, precomputed_Z_samples):
    param = jnp.maximum(param, 0.01)
    p_l, q_l, fc_l = param_to_pq0_jitted(param)
    #processed_Z = average_Z_jitted(Z)
    #del Z
    #result = get_result_jitted(p_l, q_l, fc_l, processed_Z)
    #result = get_result_monte_carlo(
    #    p_l, q_l, fc_l, processed_Z,
        #total_num_samples=100, # Fixed number of samples
        #log_std_dev_factor=0.1 # Example noise level
    #)
    result = get_mc_result_from_perturbed_Z(
        p_l, q_l, fc_l, precomputed_Z_samples,
        #total_num_samples=100, # Fixed number of samples
        #log_std_dev_factor=0.1 # Example noise level
    )
    result = -1 * result
    return result
objective0_jitted = jax.jit(objective0)

def objective0_crra(param, precomputed_Z_samples):
    param = jnp.maximum(param, 0.01)
    p_l, q_l, fc_l = param_to_pq0_jitted(param)
    #processed_Z = average_Z_jitted(Z)
    #del Z
    #result = get_result_crra_jitted(p_l, q_l, fc_l, processed_Z)
    result = get_mc_result_crra_from_perturbed_Z(
        p_l, q_l, fc_l, precomputed_Z_samples,
        #total_num_samples=100, # Fixed number of samples
        #log_std_dev_factor=0.1 # Example noise level
    )
    result = -1 * result
    return result
objective0_crra_jitted = jax.jit(objective0_crra)

def objective0_cara(param, Z):
    param = jnp.maximum(param, 0.01)
    p_l, q_l, fc_l = param_to_pq0_jitted(param)
    processed_Z = average_Z_jitted(Z)
    del Z
    result = get_result_cara_jitted(p_l, q_l, fc_l, processed_Z)
    result = -1 * result
    return result
objective0_cara_jitted = jax.jit(objective0_cara)

def objective0_quadratic(param, Z):
    param = jnp.maximum(param, 0.01)
    p_l, q_l, fc_l = param_to_pq0_jitted(param)
    #jax.debug.print("Current param {y}", y= jax.device_get(param))
    processed_Z = average_Z_jitted(Z)
    del Z
    #result_low = get_result_jitted(p_l, q_l, fc_l, Z_low, lam)
    #result_high = get_result_jitted(p_l, q_l, fc_l, Z_high, lam)
    #result = (result_low + result_high)/2
    result = get_result_quadratic_jitted(p_l, q_l, fc_l, processed_Z)
    result = -1 * result
    #result = -1 * nansum_ignore_nan_inf_jitted(result)
    #result = -1 * sum_ignore_outliers_jitted(result)
    #result_value = jax.device_get(result)
    #jax.debug.print("Current Value {x}", x= result_value)
    return result
objective0_quadratic_jitted = jax.jit(objective0_quadratic)
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
    avg_payment = r_sum_filtered/len_transactions
    result = revenue_compare_jitted(avg_payment, r0_sum_filtered/len_transactions)
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
    crra_r = crra_jitted(jnp.maximum(r, 1e-16), gamma) 
    crra_r_sum_filtered = nansum_ignore_nan_inf_jitted(crra_r)
    crra_avg_payment = crra_r_sum_filtered/len_transactions
    result = revenue_compare_jitted(crra_avg_payment,crra_r0_mean_filtered)

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

# The single Monte Carlo run logic
def _single_mc_run_conserve_core(param_pl_ql_fcl, Z_single_sample_full_array):
    """
    This contains the original core logic of get_result, but operates on a single
    perturbed Z sample. It returns the 'result'.
    """
    p_l, q_l, fc_l = param_pl_ql_fcl # Unpack parameters for clarity

    q_sum_sim = get_q_sum_sim_jitted(p_l, q_l, fc_l, Z_single_sample_full_array)
    return q_sum_sim 

@jax.jit
def get_mc_result_conserve_from_perturbed_Z(
    param: jax.Array,
    Z_perturbed_samples_batch: jax.Array # Now takes the pre-generated batch
) -> jax.Array:
    p_l, q_l, fc_l = param_to_pq0_jitted(param)
    param_for_core = (p_l, q_l, fc_l)

    # Use jax.vmap directly to process all samples in parallel
    # _single_mc_run_get_result_core is mapped over the first dimension (total_num_samples)
    monte_carlo_results_per_sample = jax.vmap(_single_mc_run_conserve_core, in_axes=(None, 0))(
        param_for_core,
        Z_perturbed_samples_batch
    )
    # Take the average of all Monte Carlo results
    average_mc_result = jnp.mean(monte_carlo_results_per_sample, axis=0)
    result=conservation_condition_jitted(average_mc_result, p_l, q_l, fc_l)
    return result


def conservation_constraint(param, Z):
    p_l, q_l, fc_l = param_to_pq0_jitted(param)
    q_sum_sim = get_q_sum_sim_jitted(p_l, q_l, fc_l, Z)
    result = conservation_condition_jitted(q_sum_sim, p_l, q_l, fc_l)
    del q_sum_sim
    return result

conservation_constraint_jitted = jax.jit(conservation_constraint)

def conservation_constraint_crra(param, Z):
    p_l, q_l, fc_l = param_to_pq0_jitted(param)
    processed_Z = average_Z_jitted(Z)
    del Z
    q_sum_sim_month = get_q_sum_sim_month_jitted(p_l, q_l, fc_l, processed_Z) # 12*sim dimension array
    result = conservation_condition_crra_jitted(q_sum_sim_month, p_l, q_l, fc_l)
    del q_sum_sim_month
    return result

conservation_constraint_crra_jitted = jax.jit(conservation_constraint_crra)

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

param0_2 = jnp.array([3.09, 5.01-3.09, 8.54-5.01, 12.9-8.54, 14.41-12.9, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    10.8-8.5, 16.5-10.8, 37-16.5
                    , 37-37
                    ])

import jax
from jax import config # <--- Add this line
import jax.numpy as jnp

Z_current = jnp.asarray(Z_current, dtype=jnp.float32)

config.update("jax_enable_x64", False)

# --- Add this block to stop any existing server ---
try:
    jax.profiler.stop_server()
    print("Stopped existing JAX profiler server.")
except ValueError:
    # This will catch the error if no server was running, which is fine
    pass
# --------------------------------------------------

jax.profiler.start_server(9999)
print("Started JAX profiler server on port 9999.")

steps = jnp.arange(-0.25, 0.3, 0.05, dtype=jnp.float32)

# Lists to store results
pl_step_agg_results = []
ql_step_agg_results = []
fcl_step_agg_results = []
r_step_agg_results = []
cs_step_agg_results = []
q_step_agg_results = []
ev_step_agg_results = []
cs_step_results = []
q_step_results = []
r_step_results = []
ev_step_results = []
error_log = []  # Store errors

param0_high = jnp.array([3, 8, 8, 8, 8, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    7.125, 7.125,7.125,7.125
                    ])

param0_med = jnp.array([3, 5, 5, 5, 5, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    7.125, 7.125,7.125,7.125
                    ])

param0_low = jnp.array([3, 3, 3, 3, 3, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    7.125, 7.125,7.125,7.125
                    ])

def get_initial_param_for_step(step: jax.Array, param_high: jax.Array, param_med: jax.Array, param_low: jax.Array) -> jax.Array:
    """
    Determines the initial parameter value based on the step according to specified ranges:
    - [-0.25, -0.15] -> param_high (inclusive)
    - (-0.15, 0.15)  -> param_med (exclusive)
    - [0.15, 0.25]   -> param_low (inclusive)

    Args:
        step: A JAX array (scalar) representing the current step value.
        param_high: JAX array for the 'high' initial parameter guess.
        param_med: JAX array for the 'medium' initial parameter guess.
        param_low: JAX array for the 'low' initial parameter guess.

    Returns:
        A JAX array representing the selected initial parameter guess.
    """
    # Define the nominal boundary points as float32 JAX arrays
    nominal_neg_15 = jnp.array(-0.15, dtype=jnp.float32)
    nominal_pos_15 = jnp.array(0.15, dtype=jnp.float32)

    # Define a small, effective tolerance for float32 comparisons.
    # np.finfo(np.float32).eps (machine epsilon) is approx 1.19e-07.
    # A small multiple of this is typically used to create a robust comparison window.
    tolerance = jnp.finfo(jnp.float32).eps * jnp.array(4.0, dtype=jnp.float32) # Using 4.0 as a small multiplier

    # --- Logic for the ranges with tolerance ---
    # Range 1: [-0.25, -0.15] -> param_high (inclusive at -0.15)
    # If step is less than or "effectively equal to" -0.15
    if step <= (nominal_neg_15 + tolerance):
        # We assume step >= -0.25 is handled by the `steps` array generation itself.
        print(f"Step {step}: Using param_high as initial guess.")
        return param_high
    # Range 2: (-0.15, 0.15) -> param_med (exclusive on both ends)
    # This means step is "effectively greater than" -0.15 AND "effectively less than" 0.15
    # The `elif` condition implicitly handles `step > (nominal_neg_15 + tolerance)`
    elif step < (nominal_pos_15 - tolerance):
        print(f"Step {step}: Using param_med as initial guess.")
        return param_med
    # Range 3: [0.15, 0.25] -> param_low (inclusive at 0.15)
    # This means step is "effectively greater than or equal to" 0.15
    else: # This catches values >= (nominal_pos_15 - tolerance)
        print(f"Step {step}: Using param_low as initial guess.")
        return param_low

# JIT compile the function
get_initial_param_for_step_jitted = jax.jit(get_initial_param_for_step)

for step in steps:
    try:
        # Update Z_step with the current step
        Z_step = Z_current.copy()  # Preserve original structure
        #Z_step = Z_history.copy()  # Preserve original structure
        Z_step = Z_current.at[:, 2].add(step)  # Modify slice (columns 3)
        Z_step = jnp.maximum(Z_step, jnp.array(1e-16, dtype=jnp.float32))
        
        precomputed_Z_samples = generate_mc_perturbed_Z_samples(Z_step)

        # Define constraints
        constraint_conserve = NonlinearConstraint(
            lambda x: get_mc_result_conserve_from_perturbed_Z(x, precomputed_Z_samples), 
            0.95, 1.0, jac='2-point', hess=BFGS()
        )
        #constraint_revenue = NonlinearConstraint(
        #    lambda x: revenue_lower_bound_constraint_jitted(x, Z_step), 
        #    0.0, jnp.inf, jac='2-point', hess=BFGS()
        #)
        
        initial_param_for_step = get_initial_param_for_step(step, param0_high, param0_med, param0_low)
        param0_high_np = np.array(initial_param_for_step, dtype=jnp.float32) # Ensure NumPy conversion

        # First optimization attempt
        solution1_nobd = cobyqa.minimize(
            lambda x: objective0_jitted(x, precomputed_Z_samples), 
            param0_high_np,
            bounds=bounds0, 
            constraints=(constraint_conserve
                         #, constraint_revenue
                         ), 
            options={'disp': False, 'feasibility_tol': 1e-6, 'radius_init': 1, 'radius_final': 0.01}
        )

        solution1_nobd.x = np.array(solution1_nobd.x)

        # Retry if optimization did not converge
        if not solution1_nobd.success:
            print(f"Step {step}: First attempt did not converge. Retrying with new initial guess.")
            solution1_nobd_2 = cobyqa.minimize(
                lambda x: objective0_jitted(x, precomputed_Z_samples), 
                np.array(solution1_nobd.x, dtype=jnp.float32),  # Ensure NumPy conversion
                bounds=bounds0, 
                constraints=(constraint_conserve
                             #, constraint_revenue
                             ), 
                options={'disp': False, 'feasibility_tol': 1e-6, 'radius_init': 1, 'radius_final': 0.01}
            )
            solution1_nobd_final = solution1_nobd_2
        else:
            print(f"Step {step}: Optimization converged successfully.")
            solution1_nobd_final = solution1_nobd

        # **Check for Constraint Violations**
        conserve_value = get_mc_result_conserve_from_perturbed_Z(solution1_nobd_final.x, precomputed_Z_samples)
        #revenue_value = revenue_lower_bound_constraint_jitted(solution1_nobd_final.x, Z_step)

        tolerance = 1e-6

        if conserve_value < (0.95 - tolerance) or conserve_value > (1.0 + tolerance):
           raise ValueError(f"Step {step}: Conservation constraint violated! Value: {conserve_value}")

        #if revenue_value < -tolerance:
        #   raise ValueError(f"Step {step}: Revenue constraint violated! Value: {revenue_value}")

        # Compute optimal price and quantities
        p_l, q_l, fc_l = param_to_pq0_jitted(solution1_nobd_final.x)

        # Process constraints
        #processed_Z = average_Z_jitted(Z_step)
        q_sum_hh_step = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_step)
        r_step = from_q_to_r_jitted(q_sum_hh_step, p_l, q_l, fc_l)
        r_step_agg = nansum_ignore_nan_inf_jitted(r_step) / 12
        cs_step = get_v_out_jitted(q_sum_hh_step, p_l, q_l, fc_l, Z_step)
        cs_step_agg = nansum_ignore_nan_inf_jitted(cs_step) / 12
        q_hh_step = q_sum_hh_step/sim
        q_step_agg = nansum_ignore_nan_inf_jitted(q_hh_step) / 12
        ev_step = get_ev_jitted(q_sum_hh_step, p_l, q_l, fc_l, Z_step)
        ev_step_agg = nansum_ignore_nan_inf_jitted(ev_step) / 12

        # Append results
        pl_step_agg_results.append(p_l)
        ql_step_agg_results.append(q_l)
        fcl_step_agg_results.append(fc_l)
        r_step_agg_results.append(r_step_agg)
        cs_step_agg_results.append(cs_step_agg)
        q_step_agg_results.append(q_step_agg)
        ev_step_agg_results.append(ev_step_agg)
        cs_step_results.append(cs_step)
        q_step_results.append(q_hh_step)
        r_step_results.append(r_step)
        ev_step_results.append(ev_step)


    except ValueError as e:
        error_log.append(str(e))  # Store error message
        print(f"Error at step {step}: {e}")  # Optional: Print errors immediately

# **Print all errors after the loop**
if error_log:
    print("\nErrors encountered during optimization:")
    for err in error_log:
        print(err)


# Convert results to arrays for further processing
pl_step_agg_results = jnp.array(pl_step_agg_results)
pl_step_agg_results  = pl_step_agg_results.T
pl_step_agg_results_df = pd.DataFrame(pl_step_agg_results)
pl_step_agg_results_df.to_csv("ramsey_price_result/price_detail_results/montecarlo_weather_avg_bound_loss05_mean_pl.csv", index=False)
del pl_step_agg_results, pl_step_agg_results_df

ql_step_agg_results = jnp.array(ql_step_agg_results)
ql_step_agg_results  = ql_step_agg_results.T
ql_step_agg_results_df = pd.DataFrame(ql_step_agg_results)
ql_step_agg_results_df.to_csv("ramsey_price_result/price_detail_results/montecarlo_weather_avg_bound_loss05_mean_ql.csv", index=False)
del ql_step_agg_results, ql_step_agg_results_df

fcl_step_agg_results = jnp.array(fcl_step_agg_results)
fcl_step_agg_results  = fcl_step_agg_results.T
fcl_step_agg_results_df = pd.DataFrame(fcl_step_agg_results)
fcl_step_agg_results_df.to_csv("ramsey_price_result/price_detail_results/montecarlo_weather_avg_bound_loss05_mean_fcl.csv", index=False)
del fcl_step_agg_results, fcl_step_agg_results_df

r_step_agg_results = jnp.array(r_step_agg_results)
r_step_agg_results_df = pd.DataFrame(r_step_agg_results)
r_step_agg_results_df.to_csv("ramsey_welfare_result/montecarlo_weather_avg_bound_loss05_mean_r.csv", index=False)

cs_step_agg_results = jnp.array(cs_step_agg_results)
cs_step_agg_results_df = pd.DataFrame(cs_step_agg_results)
cs_step_agg_results_df.to_csv("ramsey_welfare_result/montecarlo_weather_avg_bound_loss05_mean_cs.csv", index=False)

q_step_agg_results = jnp.array(q_step_agg_results)
q_step_agg_results_df = pd.DataFrame(q_step_agg_results)
q_step_agg_results_df.to_csv("ramsey_welfare_result/montecarlo_weather_avg_bound_loss05_mean_q.csv", index=False)

ev_step_agg_results = jnp.array(ev_step_agg_results)
ev_step_agg_results_df = pd.DataFrame(ev_step_agg_results)
ev_step_agg_results_df.to_csv("ramsey_welfare_result/montecarlo_weather_avg_bound_loss05_mean_ev.csv", index=False)

cs_step_results = jnp.array(cs_step_results)
cs_step_results=cs_step_results.T
cs_step_results_df = pd.DataFrame(cs_step_results)
cs_step_results_df.to_csv("ramsey_welfare_result/cs_detail_results/montecarlo_weather_avg_bound_loss05_mean_cs_steps.csv", index=False)
del cs_step_results, cs_step_results_df

q_step_results = jnp.array(q_step_results)
q_step_results=q_step_results.T
q_step_results_df = pd.DataFrame(q_step_results)
q_step_results_df.to_csv("ramsey_welfare_result/cs_detail_results/montecarlo_weather_avg_bound_loss05_mean_q_steps.csv", index=False)
del q_step_results, q_step_results_df

r_step_results = jnp.array(r_step_results)
r_step_results=r_step_results.T
r_step_results_df = pd.DataFrame(r_step_results)
r_step_results_df.to_csv("ramsey_welfare_result/cs_detail_results/montecarlo_weather_avg_bound_loss05_mean_r_steps.csv", index=False)
del r_step_results, r_step_results_df

ev_step_results = jnp.array(ev_step_results)
ev_step_results=ev_step_results.T
ev_step_results_df = pd.DataFrame(ev_step_results)
ev_step_results_df.to_csv("ramsey_welfare_result/cs_detail_results/montecarlo_weather_avg_bound_loss05_mean_ev_steps.csv", index=False)
del ev_step_results, ev_step_results_df

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

def scale_sd_non_parametric(arr, scale_factor):
    """
    Scales the spread (variability) of a non-normally distributed array
    non-parametrically by scaling deviations from the median.

    Args:
        arr (jnp.ndarray): A 1D JAX numpy array of data.
        scale_factor (float): The factor by which to scale the deviations
                               from the median.

    Returns:
        jnp.ndarray: A new array with adjusted spread.
    """
    # Ensure the input is a JAX array. If it's already jnp.ndarray, this does nothing.
    # If it's a regular numpy array, it converts it.
    if not isinstance(arr, jnp.ndarray):
        arr = jnp.array(arr)

    # Calculate the median as the central tendency
    median_val = jnp.median(arr)

    # Calculate deviations from the median
    deviations = arr - median_val

    # Scale the deviations
    scaled_deviations = deviations * scale_factor

    # Add the median back to get the scaled values
    scaled_arr = median_val + scaled_deviations

    # Handle potential negative values (clip at zero for precipitation data)
    # JAX arrays are immutable, so direct assignment like scaled_arr[condition] = value is not allowed.
    # Use jnp.where for conditional element-wise selection.
    scaled_arr = jnp.where(scaled_arr < 0, 0.0, scaled_arr) # Use 0.0 to maintain float type

    return scaled_arr

scale_sd_non_parametric_jitted = jax.jit(scale_sd_non_parametric)

s_steps = jnp.arange(0.75, 1.25+0.05, 0.05, dtype=jnp.float32)

pl_step_agg_results_iqr = []
ql_step_agg_results_iqr = []
fcl_step_agg_results_iqr = []
r_step_agg_results_iqr = []
cs_step_agg_results_iqr = []
q_step_agg_results_iqr = []
ev_step_agg_results_iqr = []
cs_step_results_iqr = []
q_step_results_iqr = []
r_step_results_iqr = []
ev_step_results_iqr = []
error_log_var = []

param0_high = jnp.array([3, 6.5, 6.5, 6.5, 6.5, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5,
                    7.125, 7.125, 7.125, 7.125
                    ])

param0_med = jnp.array([3, 5, 5, 5, 5, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5,
                    7.125, 7.125, 7.125, 7.125
                    ])

param0_low = jnp.array([3, 3, 3, 3, 3, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5,
                    7.125, 7.125, 7.125, 7.125
                    ])


def get_initial_param_for_s_step(s_step: jax.Array, 
                                 param_high: jax.Array,
                                 param_med: jax.Array, 
                                 param_low: jax.Array, 
                                 ) -> jax.Array:
    """
    Determines the initial parameter value based on the s_step according to specified ranges:
    - [0.75, 0.85] -> param_low (inclusive)
    - (0.85, 1.15) -> param_med (exclusive)
    - [1.15, 1.25] -> param_high (inclusive)

    Args:
        s_step: A JAX array (scalar) representing the current s_step value.
        param_low: JAX array for the 'low' initial parameter guess.
        param_med: JAX array for the 'medium' initial parameter guess.
        param_high: JAX array for the 'high' initial parameter guess.

    Returns:
        A JAX array representing the selected initial parameter guess.
    """
    # Define the nominal boundary points as float32 JAX arrays
    nominal_0_85 = jnp.array(0.85, dtype=jnp.float32)
    nominal_1_15 = jnp.array(1.15, dtype=jnp.float32)

    # Define a small, effective tolerance for float32 comparisons.
    # Using a small multiple of machine epsilon to create a robust comparison window.
    tolerance = jnp.finfo(jnp.float32).eps * jnp.array(4.0, dtype=jnp.float32)

    # --- Logic for the ranges with tolerance ---
    # Range 1: [0.75, 0.85] -> param_low (inclusive at 0.85)
    # If s_step is less than or "effectively equal to" 0.85
    if s_step <= (nominal_0_85 + tolerance):
        # We assume s_step >= 0.75 is handled by the `s_steps` array generation itself.
        print(f"s_Step {s_step}: Using param_low as initial guess.")
        return param_low
    # Range 2: (0.85, 1.15) -> param_med (exclusive on both ends)
    # This means s_step is "effectively greater than" 0.85 AND "effectively less than" 1.15
    # The `elif` condition implicitly handles `s_step > (nominal_0_85 + tolerance)`
    elif s_step < (nominal_1_15 - tolerance):
        print(f"s_Step {s_step}: Using param_med as initial guess.")
        return param_med
    # Range 3: [1.15, 1.25] -> param_high (inclusive at 1.15)
    # This means s_step is "effectively greater than or equal to" 1.15
    else: # This catches values >= (nominal_1_15 - tolerance)
        print(f"s_Step {s_step}: Using param_high as initial guess.")
        return param_high

for step in s_steps:
    try:
        # Update Z_step with the current step
        #Z_step = Z_history.copy()  # Preserve original structure
        Z_step = Z_current.copy()  # Preserve original structure
        Z_step = Z_step.at[:,2 ].set(scale_sd_non_parametric_jitted(Z_step[:, 2], step))
        Z_step = jnp.maximum(Z_step, jnp.array(1e-16, dtype=jnp.float32))
        
        precomputed_Z_samples = generate_mc_perturbed_Z_samples(Z_step)

        # Define constraints
        constraint_conserve = NonlinearConstraint(
            lambda x: get_mc_result_conserve_from_perturbed_Z(x, precomputed_Z_samples), 
            0.95, 1.0, jac='2-point', hess=BFGS()
        )
        #constraint_revenue = NonlinearConstraint(
        #    lambda x: revenue_lower_bound_constraint_jitted(x, Z_step), 
        #    0.0, jnp.inf, jac='2-point', hess=BFGS()
        #)
        
        ### Determine the initial value based on the current step
        # Call the function to get the initial parameter for this step
        initial_param_for_step = get_initial_param_for_s_step(step, param0_high, param0_med, param0_low)
        param0_high_np = np.array(initial_param_for_step, dtype=jnp.float32) # Ensure NumPy conversion

        # First optimization attempt
        solution1_nobd = cobyqa.minimize(
            lambda x: objective0(x, precomputed_Z_samples), 
            param0_high_np,
            bounds=bounds0, 
            constraints=(constraint_conserve
                         #, constraint_revenue
                         ), 
            options={'disp': False, 'feasibility_tol': 1e-6, 'radius_init': 1, 'radius_final': 0.01}
        )

        solution1_nobd.x = np.array(solution1_nobd.x)

        # Retry if optimization did not converge
        if not solution1_nobd.success:
            print(f"Step {step}: First attempt did not converge. Retrying with new initial guess.")
            solution1_nobd_2 = cobyqa.minimize(
                lambda x: objective0(x, precomputed_Z_samples), 
                np.array(solution1_nobd.x, dtype=jnp.float32),  # Ensure NumPy conversion
                bounds=bounds0, 
                constraints=(constraint_conserve
                             #, constraint_revenue
                             ), 
                options={'disp': False, 'feasibility_tol': 1e-6, 'radius_init': 1, 'radius_final': 0.01}
            )
            solution1_nobd_final = solution1_nobd_2
        else:
            print(f"Step {step}: Optimization converged successfully.")
            solution1_nobd_final = solution1_nobd

        # **Check for Constraint Violations**
        conserve_value = get_mc_result_conserve_from_perturbed_Z(solution1_nobd_final.x, precomputed_Z_samples)
        #revenue_value = revenue_lower_bound_constraint_jitted(solution1_nobd_final.x, Z_step)

        tolerance = 1e-6

        if conserve_value < (0.95 - tolerance) or conserve_value > (1.0 + tolerance):
           raise ValueError(f"Step {step}: Conservation constraint violated! Value: {conserve_value}")

        #if revenue_value < -tolerance:
        #   raise ValueError(f"Step {step}: Revenue constraint violated! Value: {revenue_value}")

        # Compute optimal price and quantities
        p_l, q_l, fc_l = param_to_pq0_jitted(solution1_nobd_final.x)

        # Process constraints
        #processed_Z = average_Z_jitted(Z_step)
        q_sum_hh_step = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_step)
        r_step = from_q_to_r_jitted(q_sum_hh_step, p_l, q_l, fc_l)
        r_step_agg = nansum_ignore_nan_inf_jitted(r_step) / 12
        cs_step = get_v_out_jitted(q_sum_hh_step, p_l, q_l, fc_l, Z_step)
        cs_step_agg = nansum_ignore_nan_inf_jitted(cs_step) / 12
        q_hh_step = q_sum_hh_step/sim
        q_step_agg = nansum_ignore_nan_inf_jitted(q_hh_step) / 12
        ev_step = get_ev_jitted(q_sum_hh_step, p_l, q_l, fc_l, Z_step)
        ev_step_agg = nansum_ignore_nan_inf_jitted(ev_step) / 12

        # Append results
        pl_step_agg_results_iqr.append(p_l)
        ql_step_agg_results_iqr.append(q_l)
        fcl_step_agg_results_iqr.append(fc_l)
        r_step_agg_results_iqr.append(r_step_agg)
        cs_step_agg_results_iqr.append(cs_step_agg)
        q_step_agg_results_iqr.append(q_step_agg)
        ev_step_agg_results_iqr.append(ev_step_agg)
        cs_step_results_iqr.append(cs_step)
        q_step_results_iqr.append(q_hh_step)
        r_step_results_iqr.append(r_step)
        ev_step_results_iqr.append(ev_step)

        # Clean up memory
        del q_sum_hh_step, r_step, cs_step, q_hh_step, ev_step
        gc.collect()

    except ValueError as e:
        error_log_var.append(str(e))  # Store error message
        print(f"Error at step {step}: {e}")  # Optional: Print errors immediately

# **Print all errors after the loop**
if error_log_var:
    print("\nErrors encountered during optimization:")
    for err in error_log_var:
        print(err)
        
# Convert results to arrays for further processing
pl_step_agg_results_iqr = jnp.array(pl_step_agg_results_iqr)
pl_step_agg_results_iqr  = pl_step_agg_results_iqr.T
pl_step_agg_results_iqr_df = pd.DataFrame(pl_step_agg_results_iqr)
pl_step_agg_results_iqr_df.to_csv("ramsey_price_result/price_detail_results/montecarlo_weather_avg_bound_loss05_var_pl.csv", index=False)
del pl_step_agg_results_iqr, pl_step_agg_results_iqr_df

ql_step_agg_results_iqr = jnp.array(ql_step_agg_results_iqr)
ql_step_agg_results_iqr  = ql_step_agg_results_iqr.T
ql_step_agg_results_iqr_df = pd.DataFrame(ql_step_agg_results_iqr)
ql_step_agg_results_iqr_df.to_csv("ramsey_price_result/price_detail_results/montecarlo_weather_avg_bound_loss05_var_ql.csv", index=False)
del ql_step_agg_results_iqr, ql_step_agg_results_iqr_df

fcl_step_agg_results_iqr = jnp.array(fcl_step_agg_results_iqr)
fcl_step_agg_results_iqr  = fcl_step_agg_results_iqr.T
fcl_step_agg_results_iqr_df = pd.DataFrame(fcl_step_agg_results_iqr)
fcl_step_agg_results_iqr_df.to_csv("ramsey_price_result/price_detail_results/montecarlo_weather_avg_bound_loss05_var_fcl.csv", index=False)
del fcl_step_agg_results_iqr, fcl_step_agg_results_iqr_df

r_step_agg_results_iqr = jnp.array(r_step_agg_results_iqr)
r_step_agg_results_iqr_df = pd.DataFrame(r_step_agg_results_iqr)
r_step_agg_results_iqr_df.to_csv("ramsey_welfare_result/montecarlo_weather_avg_bound_loss05_var_r.csv", index=False)

cs_step_agg_results_iqr = jnp.array(cs_step_agg_results_iqr)
cs_step_agg_results_iqr_df = pd.DataFrame(cs_step_agg_results_iqr)
cs_step_agg_results_iqr_df.to_csv("ramsey_welfare_result/montecarlo_weather_avg_bound_loss05_var_cs.csv", index=False)

q_step_agg_results_iqr = jnp.array(q_step_agg_results_iqr)
q_step_agg_results_iqr_df = pd.DataFrame(q_step_agg_results_iqr)
q_step_agg_results_iqr_df.to_csv("ramsey_welfare_result/montecarlo_weather_avg_bound_loss05_var_q.csv", index=False)

ev_step_agg_results_iqr = jnp.array(ev_step_agg_results_iqr)
ev_step_agg_results_iqr_df = pd.DataFrame(ev_step_agg_results_iqr)
ev_step_agg_results_iqr_df.to_csv("ramsey_welfare_result/montecarlo_weather_avg_bound_loss05_var_ev.csv", index=False)

cs_step_results_iqr = jnp.array(cs_step_results_iqr)
cs_step_results_iqr=cs_step_results_iqr.T
cs_step_results_iqr_df = pd.DataFrame(cs_step_results_iqr)
cs_step_results_iqr_df.to_csv("ramsey_welfare_result/cs_detail_results/montecarlo_weather_avg_bound_loss05_var_cs_steps.csv", index=False)
del cs_step_results_iqr, cs_step_results_iqr_df

q_step_results_iqr = jnp.array(q_step_results_iqr)
q_step_results_iqr=q_step_results_iqr.T
q_step_results_iqr_df = pd.DataFrame(q_step_results_iqr)
q_step_results_iqr_df.to_csv("ramsey_welfare_result/cs_detail_results/montecarlo_weather_avg_bound_loss05_var_q_steps.csv", index=False)
del q_step_results_iqr, q_step_results_iqr_df

r_step_results_iqr = jnp.array(r_step_results_iqr)
r_step_results_iqr=r_step_results_iqr.T
r_step_results_iqr_df = pd.DataFrame(r_step_results_iqr)
r_step_results_iqr_df.to_csv("ramsey_welfare_result/cs_detail_results/montecarlo_weather_avg_bound_loss05_var_r_steps.csv", index=False)
del r_step_results_iqr, r_step_results_iqr_df

ev_step_results_iqr = jnp.array(ev_step_results_iqr)
ev_step_results_iqr=ev_step_results_iqr.T
ev_step_results_iqr_df = pd.DataFrame(ev_step_results_iqr)
ev_step_results_iqr_df.to_csv("ramsey_welfare_result/cs_detail_results/montecarlo_weather_avg_bound_loss05_var_ev_steps.csv", index=False)
del ev_step_results_iqr, ev_step_results_iqr_df

# Convert results to DataFrame
results_df = pd.DataFrame({
    "steps": steps,
    "s_steps": s_steps,
    "mean_r": r_step_agg_results,
    "mean_cs": cs_step_agg_results,
    "mean_q": q_step_agg_results,
    "mean_ev": ev_step_agg_results,
    "var_r": r_step_agg_results_iqr,
    "var_cs": cs_step_agg_results_iqr,
    "var_q": q_step_agg_results_iqr,
    "var_ev": ev_step_agg_results_iqr,
})

results_df['mean_r_diff'] = (results_df['mean_r'] -r_agg_0)/r_agg_0
results_df['var_r_diff'] = (results_df['var_r'] -r_agg_0)/r_agg_0
results_df['mean_q_diff'] = (results_df['mean_q'] -q_agg_0)/q_agg_0
results_df['var_q_diff'] = (results_df['var_q'] -q_agg_0)/q_agg_0
results_df['mean_cs_diff'] = (results_df['mean_cs'] -cs_agg_0)/np.absolute(cs_agg_0)
results_df['var_cs_diff'] = (results_df['var_cs'] -cs_agg_0)/np.absolute(cs_agg_0)

results_df.to_csv("ramsey_welfare_result/montecarlo_weather_avg_bound_loss05_result.csv", index=False)

del results_df, r_step_agg_results,cs_step_agg_results,q_step_agg_results,ev_step_agg_results
del r_step_agg_results_iqr,cs_step_agg_results_iqr,q_step_agg_results_iqr,ev_step_agg_results_iqr

########################################
###### CRRA Function ##########
########################################

steps = jnp.arange(-0.25, 0.3, 0.05, dtype = jnp.float32)

# Lists to store results
pl_step_agg_results = []
ql_step_agg_results = []
fcl_step_agg_results = []
r_step_agg_results = []
cs_step_agg_results = []
q_step_agg_results = []
ev_step_agg_results = []
cs_step_results = []
q_step_results = []
r_step_results = []
ev_step_results = []
error_log = []  # Store errors

param0_high = jnp.array([3, 8, 8, 8, 8, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    7.125, 7.125,7.125,7.125
                    ])

param0_med = jnp.array([3, 5, 5, 5, 5, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    7.125, 7.125,7.125,7.125
                    ])

param0_low = jnp.array([3, 3, 3, 3, 3, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    7.125, 7.125,7.125,7.125
                    ])


def get_initial_param_for_step(step: jax.Array, param_high: jax.Array, param_med: jax.Array, param_low: jax.Array) -> jax.Array:
    """
    Determines the initial parameter value based on the step according to specified ranges:
    - [-0.25, -0.15] -> param_high (inclusive)
    - (-0.15, 0.15)  -> param_med (exclusive)
    - [0.15, 0.25]   -> param_low (inclusive)

    Args:
        step: A JAX array (scalar) representing the current step value.
        param_high: JAX array for the 'high' initial parameter guess.
        param_med: JAX array for the 'medium' initial parameter guess.
        param_low: JAX array for the 'low' initial parameter guess.

    Returns:
        A JAX array representing the selected initial parameter guess.
    """
    # Define the nominal boundary points as float32 JAX arrays
    nominal_neg_15 = jnp.array(-0.15, dtype=jnp.float32)
    nominal_pos_15 = jnp.array(0.15, dtype=jnp.float32)

    # Define a small, effective tolerance for float32 comparisons.
    # np.finfo(np.float32).eps (machine epsilon) is approx 1.19e-07.
    # A small multiple of this is typically used to create a robust comparison window.
    tolerance = jnp.finfo(jnp.float32).eps * jnp.array(4.0, dtype=jnp.float32) # Using 4.0 as a small multiplier

    # --- Logic for the ranges with tolerance ---
    # Range 1: [-0.25, -0.15] -> param_high (inclusive at -0.15)
    # If step is less than or "effectively equal to" -0.15
    if step <= (nominal_neg_15 + tolerance):
        # We assume step >= -0.25 is handled by the `steps` array generation itself.
        print(f"Step {step}: Using param_high as initial guess.")
        return param_high
    # Range 2: (-0.15, 0.15) -> param_med (exclusive on both ends)
    # This means step is "effectively greater than" -0.15 AND "effectively less than" 0.15
    # The `elif` condition implicitly handles `step > (nominal_neg_15 + tolerance)`
    elif step < (nominal_pos_15 - tolerance):
        print(f"Step {step}: Using param_med as initial guess.")
        return param_med
    # Range 3: [0.15, 0.25] -> param_low (inclusive at 0.15)
    # This means step is "effectively greater than or equal to" 0.15
    else: # This catches values >= (nominal_pos_15 - tolerance)
        print(f"Step {step}: Using param_low as initial guess.")
        return param_low


for step in steps:
    try:
        # Update Z_step with the current step
        Z_step = Z_current.copy()  # Preserve original structure
        #Z_step = Z_history.copy()  # Preserve original structure
        Z_step = Z_current.at[:, 2].add(step)  # Modify slice (columns 3)
        Z_step = jnp.maximum(Z_step, jnp.array(1e-16, dtype=jnp.float32))
        
        precomputed_Z_samples = generate_mc_perturbed_Z_samples(Z_step)

        # Define constraints
        constraint_conserve = NonlinearConstraint(
            lambda x: get_mc_result_conserve_from_perturbed_Z(x, precomputed_Z_samples),
            0.95, 1.0, jac='2-point', hess=BFGS()
        )
        #constraint_revenue = NonlinearConstraint(
        #    lambda x: revenue_lower_bound_crra_constraint_jitted(x, Z_step), 
        #    0.0, jnp.inf, jac='2-point', hess=BFGS()
        #)
        
        initial_param_for_step = get_initial_param_for_step(step, param0_high,param0_med, param0_low)
        param0_high_np = np.array(initial_param_for_step, dtype=jnp.float32) # Ensure NumPy conversion

        # First optimization attempt
        solution1_nobd = cobyqa.minimize(
            lambda x: objective0_crra(x, precomputed_Z_samples), 
            param0_high_np,
            bounds=bounds0, 
            constraints=(constraint_conserve
                         #, constraint_revenue
                         ), 
            options={'disp': False, 'feasibility_tol': 1e-6, 'radius_init': 1, 'radius_final': 0.01}
        )

        solution1_nobd.x = np.array(solution1_nobd.x)

        # Retry if optimization did not converge
        if not solution1_nobd.success:
            print(f"Step {step}: First attempt did not converge. Retrying with new initial guess.")
            solution1_nobd_2 = cobyqa.minimize(
                lambda x: objective0_crra(x, precomputed_Z_samples), 
                np.array(solution1_nobd.x, dtype=jnp.float32),  # Ensure NumPy conversion
                bounds=bounds0, 
                constraints=(constraint_conserve
                             #, constraint_revenue
                             ), 
                options={'disp': False, 'feasibility_tol': 1e-6, 'radius_init': 1, 'radius_final': 0.01}
            )
            solution1_nobd_final = solution1_nobd_2
        else:
            print(f"Step {step}: Optimization converged successfully.")
            solution1_nobd_final = solution1_nobd

        # **Check for Constraint Violations**
        conserve_value = get_mc_result_conserve_from_perturbed_Z(solution1_nobd_final.x, precomputed_Z_samples)
        #revenue_value = revenue_lower_bound_crra_constraint_jitted(solution1_nobd_final.x, Z_step)

        tolerance = 1e-6

        if conserve_value < (0.95 - tolerance) or conserve_value > (1.0 + tolerance):
           raise ValueError(f"Step {step}: Conservation constraint violated! Value: {conserve_value}")

        #if revenue_value < -tolerance:
        #   raise ValueError(f"Step {step}: Revenue constraint violated! Value: {revenue_value}")

        # Compute optimal price and quantities
        p_l, q_l, fc_l = param_to_pq0_jitted(solution1_nobd_final.x)

        # Process constraints
        #processed_Z = average_Z_jitted(Z_step)
        q_sum_hh_step = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_step)
        r_step = from_q_to_r_jitted(q_sum_hh_step, p_l, q_l, fc_l)
        r_step_agg = nansum_ignore_nan_inf_jitted(r_step) / 12
        cs_step = get_v_out_jitted(q_sum_hh_step, p_l, q_l, fc_l, Z_step)
        cs_step_agg = nansum_ignore_nan_inf_jitted(cs_step) / 12
        q_hh_step = q_sum_hh_step/sim
        q_step_agg = nansum_ignore_nan_inf_jitted(q_hh_step) / 12
        ev_step = get_ev_jitted(q_sum_hh_step, p_l, q_l, fc_l, Z_step)
        ev_step_agg = nansum_ignore_nan_inf_jitted(ev_step) / 12

        # Append results
        pl_step_agg_results.append(p_l)
        ql_step_agg_results.append(q_l)
        fcl_step_agg_results.append(fc_l)
        r_step_agg_results.append(r_step_agg)
        cs_step_agg_results.append(cs_step_agg)
        q_step_agg_results.append(q_step_agg)
        ev_step_agg_results.append(ev_step_agg)
        cs_step_results.append(cs_step)
        q_step_results.append(q_hh_step)
        r_step_results.append(r_step)
        ev_step_results.append(ev_step)

        # Clean up memory
        del q_sum_hh_step, r_step, cs_step, q_hh_step, ev_step
        gc.collect()

    except ValueError as e:
        error_log.append(str(e))  # Store error message
        print(f"Error at step {step}: {e}")  # Optional: Print errors immediately

# **Print all errors after the loop**
if error_log:
    print("\nErrors encountered during optimization:")
    for err in error_log:
        print(err)


# Convert results to arrays for further processing
pl_step_agg_results = jnp.array(pl_step_agg_results)
pl_step_agg_results  = pl_step_agg_results.T
pl_step_agg_results_df = pd.DataFrame(pl_step_agg_results)
pl_step_agg_results_df.to_csv("ramsey_price_result/price_detail_results/montecarlo_weather_gamma03_bound_loss05_mean_pl.csv", index=False)
del pl_step_agg_results, pl_step_agg_results_df

ql_step_agg_results = jnp.array(ql_step_agg_results)
ql_step_agg_results  = ql_step_agg_results.T
ql_step_agg_results_df = pd.DataFrame(ql_step_agg_results)
ql_step_agg_results_df.to_csv("ramsey_price_result/price_detail_results/montecarlo_weather_gamma03_bound_loss05_mean_ql.csv", index=False)
del ql_step_agg_results, ql_step_agg_results_df

fcl_step_agg_results = jnp.array(fcl_step_agg_results)
fcl_step_agg_results  = fcl_step_agg_results.T
fcl_step_agg_results_df = pd.DataFrame(fcl_step_agg_results)
fcl_step_agg_results_df.to_csv("ramsey_price_result/price_detail_results/montecarlo_weather_gamma03_bound_loss05_mean_fcl.csv", index=False)
del fcl_step_agg_results, fcl_step_agg_results_df

r_step_agg_results = jnp.array(r_step_agg_results)
r_step_agg_results_df = pd.DataFrame(r_step_agg_results)
r_step_agg_results_df.to_csv("ramsey_welfare_result/montecarlo_weather_gamma03_bound_loss05_mean_r.csv", index=False)

cs_step_agg_results = jnp.array(cs_step_agg_results)
cs_step_agg_results_df = pd.DataFrame(cs_step_agg_results)
cs_step_agg_results_df.to_csv("ramsey_welfare_result/montecarlo_weather_gamma03_bound_loss05_mean_cs.csv", index=False)

q_step_agg_results = jnp.array(q_step_agg_results)
q_step_agg_results_df = pd.DataFrame(q_step_agg_results)
q_step_agg_results_df.to_csv("ramsey_welfare_result/montecarlo_weather_gamma03_bound_loss05_mean_q.csv", index=False)

ev_step_agg_results = jnp.array(ev_step_agg_results)
ev_step_agg_results_df = pd.DataFrame(ev_step_agg_results)
ev_step_agg_results_df.to_csv("ramsey_welfare_result/montecarlo_weather_gamma03_bound_loss05_mean_ev.csv", index=False)

cs_step_results = jnp.array(cs_step_results)
cs_step_results=cs_step_results.T
cs_step_results_df = pd.DataFrame(cs_step_results)
cs_step_results_df.to_csv("ramsey_welfare_result/cs_detail_results/montecarlo_weather_gamma03_bound_loss05_mean_cs_steps.csv", index=False)
del cs_step_results, cs_step_results_df

q_step_results = jnp.array(q_step_results)
q_step_results=q_step_results.T
q_step_results_df = pd.DataFrame(q_step_results)
q_step_results_df.to_csv("ramsey_welfare_result/cs_detail_results/montecarlo_weather_gamma03_bound_loss05_mean_q_steps.csv", index=False)
del q_step_results, q_step_results_df

r_step_results = jnp.array(r_step_results)
r_step_results=r_step_results.T
r_step_results_df = pd.DataFrame(r_step_results)
r_step_results_df.to_csv("ramsey_welfare_result/cs_detail_results/montecarlo_weather_gamma03_bound_loss05_mean_r_steps.csv", index=False)
del r_step_results, r_step_results_df

ev_step_results = jnp.array(ev_step_results)
ev_step_results=ev_step_results.T
ev_step_results_df = pd.DataFrame(ev_step_results)
ev_step_results_df.to_csv("ramsey_welfare_result/cs_detail_results/montecarlo_weather_gamma03_bound_loss05_mean_ev_steps.csv", index=False)
del ev_step_results, ev_step_results_df

#######################################
#### Prepare Z for changing IQR ######
######################################

#### noted that the iqr in the demand model is iqr within month. This is not the focus of the research
#### The research focus on the volatility across different month both within a year

s_steps = jnp.arange(0.75, 1.25+0.05, 0.05, dtype = jnp.float32)

param0_high = jnp.array([3, 6.5, 6.5, 6.5, 6.5, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5,
                    7.125, 7.125, 7.125, 7.125
                    ])

param0_med = jnp.array([3, 5, 5, 5, 5, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5,
                    7.125, 7.125, 7.125, 7.125
                    ])

param0_low = jnp.array([3, 3, 3, 3, 3, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5,
                    7.125, 7.125, 7.125, 7.125
                    ])

def get_initial_param_for_s_step(s_step: jax.Array, 
                                 param_high: jax.Array,
                                 param_med: jax.Array, 
                                 param_low: jax.Array, 
                                 ) -> jax.Array:
    """
    Determines the initial parameter value based on the s_step according to specified ranges:
    - [0.75, 0.85] -> param_low (inclusive)
    - (0.85, 1.15) -> param_med (exclusive)
    - [1.15, 1.25] -> param_high (inclusive)

    Args:
        s_step: A JAX array (scalar) representing the current s_step value.
        param_low: JAX array for the 'low' initial parameter guess.
        param_med: JAX array for the 'medium' initial parameter guess.
        param_high: JAX array for the 'high' initial parameter guess.

    Returns:
        A JAX array representing the selected initial parameter guess.
    """
    # Define the nominal boundary points as float32 JAX arrays
    nominal_0_85 = jnp.array(0.85, dtype=jnp.float32)
    nominal_1_15 = jnp.array(1.15, dtype=jnp.float32)

    # Define a small, effective tolerance for float32 comparisons.
    # Using a small multiple of machine epsilon to create a robust comparison window.
    tolerance = jnp.finfo(jnp.float32).eps * jnp.array(4.0, dtype=jnp.float32)

    # --- Logic for the ranges with tolerance ---
    # Range 1: [0.75, 0.85] -> param_low (inclusive at 0.85)
    # If s_step is less than or "effectively equal to" 0.85
    if s_step <= (nominal_0_85 + tolerance):
        # We assume s_step >= 0.75 is handled by the `s_steps` array generation itself.
        print(f"s_Step {s_step}: Using param_low as initial guess.")
        return param_low
    # Range 2: (0.85, 1.15) -> param_med (exclusive on both ends)
    # This means s_step is "effectively greater than" 0.85 AND "effectively less than" 1.15
    # The `elif` condition implicitly handles `s_step > (nominal_0_85 + tolerance)`
    elif s_step < (nominal_1_15 - tolerance):
        print(f"s_Step {s_step}: Using param_med as initial guess.")
        return param_med
    # Range 3: [1.15, 1.25] -> param_high (inclusive at 1.15)
    # This means s_step is "effectively greater than or equal to" 1.15
    else: # This catches values >= (nominal_1_15 - tolerance)
        print(f"s_Step {s_step}: Using param_high as initial guess.")
        return param_high

pl_step_agg_results_iqr = []
ql_step_agg_results_iqr = []
fcl_step_agg_results_iqr = []
r_step_agg_results_iqr = []
cs_step_agg_results_iqr = []
q_step_agg_results_iqr = []
ev_step_agg_results_iqr = []
cs_step_results_iqr = []
q_step_results_iqr = []
r_step_results_iqr = []
ev_step_results_iqr = []
error_log_var = []

for step in s_steps:
    try:
        # Update Z_step with the current step
        #Z_step = Z_history.copy()  # Preserve original structure
        Z_step = Z_current.copy()  # Preserve original structure
        Z_step = Z_step.at[:,2 ].set(scale_sd_non_parametric_jitted(Z_step[:, 2], step))
        Z_step = jnp.maximum(Z_step, jnp.array(1e-16, dtype=jnp.float32))
        
        precomputed_Z_samples = generate_mc_perturbed_Z_samples(Z_step)

        # Define constraints
        constraint_conserve = NonlinearConstraint(
            lambda x: get_mc_result_conserve_from_perturbed_Z(x, precomputed_Z_samples), 
            0.95, 1.0, jac='2-point', hess=BFGS()
        )
        #constraint_revenue = NonlinearConstraint(
        #    lambda x: revenue_lower_bound_crra_constraint_jitted(x, Z_step), 
        #    0.0, jnp.inf, jac='2-point', hess=BFGS()
        #)
        
        initial_param_for_step = get_initial_param_for_s_step(step, param0_high, param0_med, param0_low)
        param0_high_np = np.array(initial_param_for_step, dtype=jnp.float32) # Ensure NumPy conversion

        # First optimization attempt
        solution1_nobd = cobyqa.minimize(
            lambda x: objective0_crra(x, precomputed_Z_samples), 
            param0_high_np,
            bounds=bounds0, 
            constraints=(constraint_conserve
                         #, constraint_revenue
                         ), 
            options={'disp': False, 'feasibility_tol': 1e-6, 'radius_init': 1, 'radius_final': 0.01}
        )

        solution1_nobd.x = np.array(solution1_nobd.x)

        # Retry if optimization did not converge
        if not solution1_nobd.success:
            print(f"Step {step}: First attempt did not converge. Retrying with new initial guess.")
            solution1_nobd_2 = cobyqa.minimize(
                lambda x: objective0_crra(x, precomputed_Z_samples), 
                np.array(solution1_nobd.x, dtype=jnp.float32),  # Ensure NumPy conversion
                bounds=bounds0, 
                constraints=(constraint_conserve
                             #, constraint_revenue
                             ), 
                options={'disp': False, 'feasibility_tol': 1e-6, 'radius_init': 1, 'radius_final': 0.01}
            )
            solution1_nobd_final = solution1_nobd_2
        else:
            print(f"Step {step}: Optimization converged successfully.")
            solution1_nobd_final = solution1_nobd

        # **Check for Constraint Violations**
        conserve_value = get_mc_result_conserve_from_perturbed_Z(solution1_nobd_final.x, precomputed_Z_samples)
        #revenue_value = revenue_lower_bound_crra_constraint_jitted(solution1_nobd_final.x, Z_step)
        
        tolerance = 1e-6

        if conserve_value < (0.95 - tolerance) or conserve_value > (1.0 + tolerance):
           raise ValueError(f"Step {step}: Conservation constraint violated! Value: {conserve_value}")

        #if revenue_value < -tolerance:
        #   raise ValueError(f"Step {step}: Revenue constraint violated! Value: {revenue_value}")

        # Compute optimal price and quantities
        p_l, q_l, fc_l = param_to_pq0_jitted(solution1_nobd_final.x)

        # Process constraints
        #processed_Z = average_Z_jitted(Z_step)
        q_sum_hh_step = get_q_sum_hh_jitted(p_l, q_l, fc_l, Z_step)
        r_step = from_q_to_r_jitted(q_sum_hh_step, p_l, q_l, fc_l)
        r_step_agg = nansum_ignore_nan_inf_jitted(r_step) / 12
        cs_step = get_v_out_jitted(q_sum_hh_step, p_l, q_l, fc_l, Z_step)
        cs_step_agg = nansum_ignore_nan_inf_jitted(cs_step) / 12
        q_hh_step = q_sum_hh_step/sim
        q_step_agg = nansum_ignore_nan_inf_jitted(q_hh_step) / 12
        ev_step = get_ev_jitted(q_sum_hh_step, p_l, q_l, fc_l, Z_step)
        ev_step_agg = nansum_ignore_nan_inf_jitted(ev_step) / 12

        # Append results
        pl_step_agg_results_iqr.append(p_l)
        ql_step_agg_results_iqr.append(q_l)
        fcl_step_agg_results_iqr.append(fc_l)
        r_step_agg_results_iqr.append(r_step_agg)
        cs_step_agg_results_iqr.append(cs_step_agg)
        q_step_agg_results_iqr.append(q_step_agg)
        ev_step_agg_results_iqr.append(ev_step_agg)
        cs_step_results_iqr.append(cs_step)
        q_step_results_iqr.append(q_hh_step)
        r_step_results_iqr.append(r_step)
        ev_step_results_iqr.append(ev_step)

        # Clean up memory
        del q_sum_hh_step, r_step, cs_step, q_hh_step, ev_step
        gc.collect()

    except ValueError as e:
        error_log_var.append(str(e))  # Store error message
        print(f"Error at step {step}: {e}")  # Optional: Print errors immediately

# **Print all errors after the loop**
if error_log_var:
    print("\nErrors encountered during optimization:")
    for err in error_log_var:
        print(err)
             

# Convert results to arrays for further processing
pl_step_agg_results_iqr = jnp.array(pl_step_agg_results_iqr)
pl_step_agg_results_iqr  = pl_step_agg_results_iqr.T
pl_step_agg_results_iqr_df = pd.DataFrame(pl_step_agg_results_iqr)
pl_step_agg_results_iqr_df.to_csv("ramsey_price_result/price_detail_results/montecarlo_weather_gamma03_bound_loss05_var_pl.csv", index=False)
del pl_step_agg_results_iqr, pl_step_agg_results_iqr_df

ql_step_agg_results_iqr = jnp.array(ql_step_agg_results_iqr)
ql_step_agg_results_iqr  = ql_step_agg_results_iqr.T
ql_step_agg_results_iqr_df = pd.DataFrame(ql_step_agg_results_iqr)
ql_step_agg_results_iqr_df.to_csv("ramsey_price_result/price_detail_results/montecarlo_weather_gamma03_bound_loss05_var_ql.csv", index=False)
del ql_step_agg_results_iqr, ql_step_agg_results_iqr_df

fcl_step_agg_results_iqr = jnp.array(fcl_step_agg_results_iqr)
fcl_step_agg_results_iqr  = fcl_step_agg_results_iqr.T
fcl_step_agg_results_iqr_df = pd.DataFrame(fcl_step_agg_results_iqr)
fcl_step_agg_results_iqr_df.to_csv("ramsey_price_result/price_detail_results/montecarlo_weather_gamma03_bound_loss05_var_fcl.csv", index=False)
del fcl_step_agg_results_iqr, fcl_step_agg_results_iqr_df

r_step_agg_results_iqr = jnp.array(r_step_agg_results_iqr)
r_step_agg_results_iqr_df = pd.DataFrame(r_step_agg_results_iqr)
r_step_agg_results_iqr_df.to_csv("ramsey_welfare_result/montecarlo_weather_gamma03_bound_loss05_var_r.csv", index=False)

cs_step_agg_results_iqr = jnp.array(cs_step_agg_results_iqr)
cs_step_agg_results_iqr_df = pd.DataFrame(cs_step_agg_results_iqr)
cs_step_agg_results_iqr_df.to_csv("ramsey_welfare_result/montecarlo_weather_gamma03_bound_loss05_var_cs.csv", index=False)

q_step_agg_results_iqr = jnp.array(q_step_agg_results_iqr)
q_step_agg_results_iqr_df = pd.DataFrame(q_step_agg_results_iqr)
q_step_agg_results_iqr_df.to_csv("ramsey_welfare_result/montecarlo_weather_gamma03_bound_loss05_var_q.csv", index=False)

ev_step_agg_results_iqr = jnp.array(ev_step_agg_results_iqr)
ev_step_agg_results_iqr_df = pd.DataFrame(ev_step_agg_results_iqr)
ev_step_agg_results_iqr_df.to_csv("ramsey_welfare_result/montecarlo_weather_gamma03_bound_loss05_var_ev.csv", index=False)

cs_step_results_iqr = jnp.array(cs_step_results_iqr)
cs_step_results_iqr=cs_step_results_iqr.T
cs_step_results_iqr_df = pd.DataFrame(cs_step_results_iqr)
cs_step_results_iqr_df.to_csv("ramsey_welfare_result/cs_detail_results/montecarlo_weather_gamma03_bound_loss05_var_cs_steps.csv", index=False)
del cs_step_results_iqr, cs_step_results_iqr_df

q_step_results_iqr = jnp.array(q_step_results_iqr)
q_step_results_iqr=q_step_results_iqr.T
q_step_results_iqr_df = pd.DataFrame(q_step_results_iqr)
q_step_results_iqr_df.to_csv("ramsey_welfare_result/cs_detail_results/montecarlo_weather_gamma03_bound_loss05_var_q_steps.csv", index=False)
del q_step_results_iqr, q_step_results_iqr_df

r_step_results_iqr = jnp.array(r_step_results_iqr)
r_step_results_iqr=r_step_results_iqr.T
r_step_results_iqr_df = pd.DataFrame(r_step_results_iqr)
r_step_results_iqr_df.to_csv("ramsey_welfare_result/cs_detail_results/montecarlo_weather_gamma03_bound_loss05_var_r_steps.csv", index=False)
del r_step_results_iqr, r_step_results_iqr_df

ev_step_results_iqr = jnp.array(ev_step_results_iqr)
ev_step_results_iqr=ev_step_results_iqr.T
ev_step_results_iqr_df = pd.DataFrame(ev_step_results_iqr)
ev_step_results_iqr_df.to_csv("ramsey_welfare_result/cs_detail_results/montecarlo_weather_gamma03_bound_loss05_var_ev_steps.csv", index=False)
del ev_step_results_iqr, ev_step_results_iqr_df

# Convert results to DataFrame
results_df = pd.DataFrame({
    "steps": steps,
    "s_steps": s_steps,
    "mean_r": r_step_agg_results,
    "mean_cs": cs_step_agg_results,
    "mean_q": q_step_agg_results,
    "mean_ev": ev_step_agg_results,
    "var_r": r_step_agg_results_iqr,
    "var_cs": cs_step_agg_results_iqr,
    "var_q": q_step_agg_results_iqr,
    "var_ev": ev_step_agg_results_iqr,
})

results_df['mean_r_diff'] = (results_df['mean_r'] -r_agg_0)/r_agg_0
results_df['var_r_diff'] = (results_df['var_r'] -r_agg_0)/r_agg_0
results_df['mean_q_diff'] = (results_df['mean_q'] -q_agg_0)/q_agg_0
results_df['var_q_diff'] = (results_df['var_q'] -q_agg_0)/q_agg_0
results_df['mean_cs_diff'] = (results_df['mean_cs'] -cs_agg_0)/np.absolute(cs_agg_0)
results_df['var_cs_diff'] = (results_df['var_cs'] -cs_agg_0)/np.absolute(cs_agg_0)

results_df.to_csv("ramsey_welfare_result/montecarlo_weather_gamma03_bound_loss05_result.csv", index=False)
