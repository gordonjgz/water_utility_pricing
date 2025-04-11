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

#def find_first_nonnegative(p):
    #def cond_fun(state):
      #  idx, found, p = state
     #   return jnp.logical_and(idx < p.size, jnp.logical_not(found))

    #def body_fun(state):
       # idx, found, p = state
       # found = jnp.logical_or(found, p[idx] >= 0)
      #  idx = idx + 1
     #   return idx, found, p

    #init_state = (0, False, p)
   # idx, found, _ = lax.while_loop(cond_fun, body_fun, init_state)
  #  
 #   return jnp.where(found, idx - 1, 5)

#find_first_nonnegative_jit = jit(find_first_nonnegative)

#def mean_ignore_nan_inf(x):
 #   return np.nanmean(x.replace([np.inf, -np.inf], np.nan))

 
def calculate_log_w(p_l, q_l, fc_l):
    #fc_l = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.25+29.75])
    q_kink_l =q_l
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

def mean_ignore_na_inf(series):
    # Drop NaN values
    series = series.dropna()
    # Filter out infinite values
    series = series[np.isfinite(series)]
    # Calculate the mean of the remaining values
    return series.mean()

mean_ignore_na_inf_jitted = jax.jit(mean_ignore_na_inf)

def safe_std(x):
    clean_x = x[~x.isin([jnp.nan, jnp.inf, -jnp.inf])]
    return clean_x.std()
safe_syd_jitted = jax.jit(safe_std)

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

################################################
#### Calcualte eta for each household ##########
################################################

def get_eta (p_l, q_l, fc_l, d_id, quantity):
    log_w_k_base_0 = calculate_log_w_jitted(p_l, q_l, fc_l)
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
    diff_sim = jnp.log(jnp.array(demand_2018_small['quantity'])[:, jnp.newaxis]) - log_w_base_0_sim
    diff_hat = jnp.mean(diff_sim, axis=1)
    demand_2018_small.loc[:, 'e_diff'] = diff_hat
    demand_2018_small =  demand_2018_small.replace([jnp.inf, -jnp.inf], jnp.nan).dropna(subset=['e_diff'])
    demand_2018_small['mean_e_diff'] = demand_2018_small.groupby('prem_id')['e_diff'].transform('mean')
    demand_2018_small['sigma_e_diff'] = demand_2018_small.groupby('prem_id')['e_diff'].transform('std')
    #demand_2018_small = demand_2018_small.sort_values(by=['prem_id', 'bill_ym'])
    return demand_2018_small

get_eta_jitted = jax.jit(get_eta)


#demand_2018_using = demand_2018_using[pd.notna(demand_2018_using['e_eta']) & np.isfinite(demand_2018_using['e_eta'])]

#demand_2018_using_eta = demand_2018_using [['prem_id', 'bill_ym', 'e_alpha','e_eta','mean_e_eta', 'sigma_e_eta']]

#demand_2018_using_eta = demand_2018_using_eta.sort_values(by=['prem_id', 'bill_ym'])
p_l0 = jnp.array([ 3.09,  5.01,  8.54, 12.9 , 14.41])
q_l0 = jnp.array([ 2,  6, 11, 20])
fc_l0 = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.25+29.75])

demand_2018_id = demand_2018_using[['prem_id', 'bill_ym']]
demand_2018_using_eta = get_eta(p_l0, q_l0, fc_l0, 
                                       demand_2018_id, 
                                       jnp.array(demand_2018_using['quantity']))

sigma_eta_df_sum = pd.DataFrame(demand_2018_using_eta.groupby('prem_id')[['mean_e_diff', 'sigma_e_diff']].mean())
sigma_eta_df_sum = sigma_eta_df_sum.groupby('prem_id').sum().reset_index()


demand_2018_using_new = pd.merge(demand_2018_using, sigma_eta_df_sum, on='prem_id')

demand_2018_using_eta = demand_2018_using_new[['prem_id', 'bill_ym', 'quantity', 'mean_e_diff', 'sigma_e_diff']]

demand_2018_using_eta.to_csv('demand_2018_using_eta.csv', index=False)

demand_2018_using_new.to_csv('demand_2018_using_new.csv', index=False)

###############
## End Set up
###############

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

p_l0 = jnp.array([ 3.09,  5.01,  8.54, 12.9 , 14.41])
q_l0 = jnp.array([ 2,  6, 11, 20])
fc_l0 = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.25+29.75])

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

A_current_outdoor = jnp.column_stack(( 
    jnp.array(demand_2018_using_new['heavy_water_app_area']), 
                                      jnp.array(demand_2018_using_new['lawn_areaxNDVI']), 
                                      jnp.array(demand_2018_using_new['above_one_acre'])
                                      ))
A_current_indoor = jnp.column_stack((jnp.array(demand_2018_using_new['bathroom']),
                                     jnp.array(demand_2018_using_new['above_one_acre'])
                                       ))
A_current = jnp.column_stack((
    jnp.array(demand_2018_using_new['heavy_water_app']),
    jnp.array(demand_2018_using_new['lawn_areaxNDVI']),
    jnp.array(demand_2018_using_new['bathroom'])
    ))
Z_current_outdoor = jnp.column_stack((jnp.array(demand_2018_using_new['mean_TMAX_1']),
                                      jnp.array(demand_2018_using_new['IQR_TMAX_1']),
                                      jnp.array(demand_2018_using_new['total_PRCP']) 
                                      ,jnp.array(demand_2018_using_new['IQR_PRCP'])))
Z_current_indoor = jnp.array(demand_2018_using_new['mean_TMAX_1'])
Z_current_indoor = Z_current_indoor[:, jnp.newaxis]
G = jnp.array(demand_2018_using_new['previous_NDVImyd_diff'])
I = jnp.array(demand_2018_using_new['income'])
p0 = jnp.array(demand_2018_using_new['previous_essential_usage_mp'])
w_i = jnp.array(demand_2018_using_new['quantity'])
de = jnp.array(demand_2018_using_new['deflator'])
q_statusquo = jnp.array(demand_2018_using_new['quantity'])

q_statusquo_sum = jnp.sum(q_statusquo)

#del demand_2018_using_new

def calculate_log_w(p_l, q_l, fc_l):
    #fc_l = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.25+29.75])
    q_kink_l =q_l
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
                  G = G,
                  p = p_l, I = I,
                  p0 =p0, 
                  de = de,
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

demand_2018_using_eta_unique = demand_2018_using_eta.drop(columns=['bill_ym', 'quantity']).drop_duplicates().reset_index()
#demand_2018_using_eta_unique = demand_2018_using_eta.drop(columns=['bill_ym']).drop_duplicates()
np.random.seed(42)
sim = 100
#key = random.PRNGKey(42)
random_data = np.random.normal(loc=demand_2018_using_eta_unique['mean_e_diff'].values[:, None]
                                   , scale=sigma_eta, size=(demand_2018_using_eta_unique.shape[0], sim))
#random_data = np.random.normal(loc=demand_2018_using_eta_unique['mean_e_eta'].values[:, None]
 #                                  , scale=demand_2018_using_eta_unique['sigma_e_eta'].values[:, None], size=(demand_2018_using_eta_unique.shape[0], sim))
random_df = pd.DataFrame(random_data, columns=[f'Eta_{i+1}' for i in range(sim)]).reset_index()

sigma_eta_df_sum = pd.concat([demand_2018_using_eta_unique, random_df], axis=1)
sigma_eta_df_new = pd.merge(demand_2018_using_eta, sigma_eta_df_sum, on='prem_id', how='left')   
shape = (sim, 1)  
#nu_array = sigma_nu * random.normal(key, shape)
nu_array = np.random.normal(loc = 0, scale = sigma_nu, size = shape)
nu_array = jnp.minimum(nu_array, 7)
#nu_array = jnp.maximum(nu_array, -7)
eta_l = jnp.array(sigma_eta_df_new .iloc[:, -sim:])
eta_l = jnp.minimum(eta_l, 7)
eta_l = jnp.maximum(eta_l, -7)
len_transactions = len( demand_2018_using_eta) 

del demand_2018_using_eta_unique, random_data, random_df, sigma_eta_df_sum, sigma_eta_df_new

def gen_nu_array(sigma_mu):
    #shape = (sim, 1)
    shape = (len_transactions, sim)
    #nu_array = sigma_nu * random.normal(key, shape)
    nu_array = np.random.normal(loc = 0, scale = sigma_nu, size = shape)
    nu_array = jnp.minimum(nu_array, 2)
    nu_array = jnp.maximum(nu_array, -2)
    return nu_array
gen_nu_array_jitted = jax.jit(gen_nu_array)

def get_log_q_inner(log_w_k, n, q_l):
    log_w1 = log_w_k[0]
    log_w2 = log_w_k[1]
    log_w3 = log_w_k[2]
    log_w4 = log_w_k[3]
    log_w5 = log_w_k[4]
    e = log_w_k[-sim:]
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

def cf_w (p_l, q_l, fc_l, 
          #nu_array = nu_array,
          eta_l = eta_l):
    log_q_sim = get_log_q_sim_jitted(p_l, q_l, fc_l)
    log_q_sim = log_q_sim.reshape(len_transactions, sim)
    #gc.collect()
    nu_array =  gen_nu_array_jitted(sigma_nu)
    return log_q_sim + nu_array
cf_w_jitted = jax.jit(cf_w)


def get_log_w(p_l, q_l, fc_l, 
          #nu_array = nu_array,
          eta_l = eta_l):
    log_w = calculate_log_w_jitted(p_l, q_l, fc_l)
    log_w = jnp.column_stack((log_w, eta_l))
    return log_w
get_log_w_jitted = jax.jit(get_log_w)

def get_log_q_sim(p_l, q_l, fc_l, 
          #nu_array = nu_array,
          eta_l = eta_l):
    #nu_array = gen_nu_array_jitted(sigma_nu)
    log_w = get_log_w_jitted(p_l, q_l, fc_l, eta_l = eta_l)
    
    def get_log_q (log_w_k, n = 0, q_l = q_l):
        #log_q_nonu = get_log_q_inner_jitted(log_w_k, n, q_l)
        #nu_array = gen_nu_array_jitted(sigma_nu)
        return get_log_q_inner_jitted(log_w_k, n, q_l)
    get_log_q_jitted = jax.jit(get_log_q)
    
    log_q_sim = jnp.apply_along_axis(get_log_q_jitted, axis=1, arr = log_w)
    #nu_array = gen_nu_array_jitted(sigma_nu)
    return log_q_sim
get_log_q_sim_jitted = jax.jit(get_log_q_sim)

log_q0 = cf_w_jitted(p_l0, q_l0, fc_l0)

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

q0_df['q_diff'] = q0_df['q_sim_mean'] - q0_df['q_statusquo'] 
q0_df['q_diff_perct'] = (q0_df['q_sim_mean'] - q0_df['q_statusquo'] )/q0_df['q_statusquo']

q0_df = q0_df[q0_df['q_diff_perct'] <= 5]
q0_df_key = q0_df[['prem_id', 'bill_ym']]

#demand_2018_using_new = pd.read_csv('demand_2018_using_new.csv')

demand_2018_using_new = pd.merge(demand_2018_using_new, q0_df_key,on=['prem_id', 'bill_ym'], how='right')

#demand_2018_using_eta = pd.read_csv('demand_2018_using_eta.csv')
demand_2018_using_eta = pd.merge(demand_2018_using_eta, q0_df_key,on=['prem_id', 'bill_ym'], how='right')

demand_2018_using_eta.to_csv('demand_2018_using_eta.csv', index=False)

demand_2018_using_new.to_csv('demand_2018_using_new.csv', index=False)


demand_2018_using_eta = pd.read_csv('demand_2018_using_eta.csv')

demand_2018_using_new = pd.read_csv('demand_2018_using_new.csv')

####################################################
###### Ending Second Set up ########
##########################################################

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

p_l0 = jnp.array([ 3.09,  5.01,  8.54, 12.9 , 14.41])
q_l0 = jnp.array([ 2,  6, 11, 20])
fc_l0 = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.25+29.75])

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

A_current_outdoor = jnp.column_stack(( 
    jnp.array(demand_2018_using_new['heavy_water_app_area']), 
                                      jnp.array(demand_2018_using_new['lawn_areaxNDVI']), 
                                      jnp.array(demand_2018_using_new['above_one_acre'])
                                      ))
A_current_indoor = jnp.column_stack((jnp.array(demand_2018_using_new['bathroom']),
                                     jnp.array(demand_2018_using_new['above_one_acre'])
                                       ))
A_current = jnp.column_stack((
    jnp.array(demand_2018_using_new['heavy_water_app']),
    jnp.array(demand_2018_using_new['lawn_areaxNDVI']),
    jnp.array(demand_2018_using_new['bathroom'])
    ))
Z_current_outdoor = jnp.column_stack((jnp.array(demand_2018_using_new['mean_TMAX_1']),
                                      jnp.array(demand_2018_using_new['IQR_TMAX_1']),
                                      jnp.array(demand_2018_using_new['total_PRCP']) 
                                      ,jnp.array(demand_2018_using_new['IQR_PRCP'])))
Z_current_indoor = jnp.array(demand_2018_using_new['mean_TMAX_1'])
Z_current_indoor = Z_current_indoor[:, jnp.newaxis]
G = jnp.array(demand_2018_using_new['previous_NDVImyd_diff'])
I = jnp.array(demand_2018_using_new['income'])
p0 = jnp.array(demand_2018_using_new['previous_essential_usage_mp'])
w_i = jnp.array(demand_2018_using_new['quantity'])
de = jnp.array(demand_2018_using_new['deflator'])
q_statusquo = jnp.array(demand_2018_using_new['quantity'])

q_statusquo_sum = jnp.sum(q_statusquo)

def calculate_log_w(p_l, q_l, fc_l):
    #fc_l = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.25+29.75])
    q_kink_l =q_l
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
                  G = G,
                  p = p_l, I = I,
                  p0 =p0, 
                  de = de,
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
demand_2018_using_eta_unique = demand_2018_using_eta.drop(columns=['bill_ym', 'quantity']).drop_duplicates().reset_index()
#demand_2018_using_eta_unique = demand_2018_using_eta.drop(columns=['bill_ym']).drop_duplicates()
np.random.seed(42)
sim = 100
#key = random.PRNGKey(42)

random_data = np.random.normal(loc=demand_2018_using_eta_unique['mean_e_diff'].values[:, None]
                                   , scale=sigma_eta, size=(demand_2018_using_eta_unique.shape[0], sim))
#random_data = np.random.normal(loc=demand_2018_using_eta_unique['mean_e_eta'].values[:, None]
 #                                  , scale=demand_2018_using_eta_unique['sigma_e_eta'].values[:, None], size=(demand_2018_using_eta_unique.shape[0], sim))
random_df = pd.DataFrame(random_data, columns=[f'Eta_{i+1}' for i in range(sim)]).reset_index()

sigma_eta_df_sum = pd.concat([demand_2018_using_eta_unique, random_df], axis=1)
sigma_eta_df_new = pd.merge(demand_2018_using_eta, sigma_eta_df_sum, on='prem_id', how='left')   
shape = (sim, 1)  
#nu_array = sigma_nu * random.normal(key, shape)
nu_array = np.random.normal(loc = 0, scale = sigma_nu, size = shape)
nu_array = jnp.minimum(nu_array, 7)
#nu_array = jnp.maximum(nu_array, -7)
eta_l = jnp.array(sigma_eta_df_new .iloc[:, -sim:])
eta_l = jnp.minimum(eta_l, 7)
eta_l = jnp.maximum(eta_l, -7)
len_transactions = len( demand_2018_using_eta) 

del demand_2018_using_eta_unique, random_data, random_df, sigma_eta_df_sum, sigma_eta_df_new

def gen_nu_array(sigma_mu):
    #shape = (sim, 1)
    shape = (len_transactions, sim)
    #nu_array = sigma_nu * random.normal(key, shape)
    nu_array = np.random.normal(loc = 0, scale = sigma_nu, size = shape)
    nu_array = jnp.minimum(nu_array, 2)
    nu_array = jnp.maximum(nu_array, -2)
    return nu_array
gen_nu_array_jitted = jax.jit(gen_nu_array)

def get_log_q_inner(log_w_k, n, q_l):
    log_w1 = log_w_k[0]
    log_w2 = log_w_k[1]
    log_w3 = log_w_k[2]
    log_w4 = log_w_k[3]
    log_w5 = log_w_k[4]
    e = log_w_k[-sim:]
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

def cf_w (p_l, q_l, fc_l, 
          #nu_array = nu_array,
          eta_l = eta_l):
    log_q_sim = get_log_q_sim_jitted(p_l, q_l, fc_l)
    log_q_sim = log_q_sim.reshape(len_transactions, sim)
    #gc.collect()
    nu_array =  gen_nu_array_jitted(sigma_nu)
    return log_q_sim + nu_array
cf_w_jitted = jax.jit(cf_w)


def get_log_w(p_l, q_l, fc_l, 
          #nu_array = nu_array,
          eta_l = eta_l):
    log_w = calculate_log_w_jitted(p_l, q_l, fc_l)
    log_w = jnp.column_stack((log_w, eta_l))
    return log_w
get_log_w_jitted = jax.jit(get_log_w)

def get_log_q_sim(p_l, q_l, fc_l, 
          #nu_array = nu_array,
          eta_l = eta_l):
    #nu_array = gen_nu_array_jitted(sigma_nu)
    log_w = get_log_w_jitted(p_l, q_l, fc_l, eta_l = eta_l)
    
    def get_log_q (log_w_k, n = 0, q_l = q_l):
        #log_q_nonu = get_log_q_inner_jitted(log_w_k, n, q_l)
        #nu_array = gen_nu_array_jitted(sigma_nu)
        return get_log_q_inner_jitted(log_w_k, n, q_l)
    get_log_q_jitted = jax.jit(get_log_q)
    
    log_q_sim = jnp.apply_along_axis(get_log_q_jitted, axis=1, arr = log_w)
    #nu_array = gen_nu_array_jitted(sigma_nu)
    return log_q_sim
get_log_q_sim_jitted = jax.jit(get_log_q_sim)

log_q0 = cf_w_jitted(p_l0, q_l0, fc_l0)

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

q0_df['q_diff'] = q0_df['q_sim_mean'] - q0_df['q_statusquo'] 
q0_df['q_diff_perct'] = (q0_df['q_sim_mean'] - q0_df['q_statusquo'] )/q0_df['q_statusquo']

#q0_sum_sum = jnp.sum(q0_sum)


#### So far the prediction of q0 is still too big. The probable reason is the demand estimation needs to be improved. 



def cf_w_moment(p_l, q_l, fc_l, sigma_eta_df):
    log_q = cf_w(p_l, q_l, fc_l,sigma_eta_df)
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

moment_0_mean = cf_w_moment_mean(log_q0, p_l0, q_l0, fc_l0)

r0 = jnp.array(moment_0_mean['mean_e_r'])[:,jnp.newaxis]
del moment_0_mean


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

hh_size = len(np.unique(np.array(demand_2018_using_eta['prem_id'], dtype = np.int64)))
sim = 100

def parsing():
    pass

def nansum_ignore_nan_inf_otheraxis(arr):
    mask = jnp.logical_and(jnp.isfinite(arr), ~jnp.isnan(arr))  # Mask out inf and NaN
    return jnp.sum(jnp.where(mask, arr, 0), axis = 1)
nansum_ignore_nan_inf_otheraxis_jitted = jax.jit(nansum_ignore_nan_inf_otheraxis)

def get_q_sum(p_l, q_l, fc_l):
    log_q = cf_w_jitted(p_l, q_l, fc_l)
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

def get_q_sum_hh(p_l, q_l, fc_l):
    log_q = cf_w_jitted(p_l, q_l, fc_l)
    chunk_size = 100 
    num_columns = log_q.shape[1]
    result_hh = []
    #result_sim = []
    for start_col in range(0, num_columns, chunk_size):
        end_col = min(start_col + chunk_size, num_columns)
        data_chunk = log_q[:, start_col:end_col]
        result_chunk = jnp.exp(data_chunk)
        result_chunk_hh = nansum_ignore_nan_inf_otheraxis_jitted(result_chunk)
        result_hh.append(result_chunk_hh)
        #result_chunk_sim = nansum_ignore_nan_inf_jitted(result_chunk)
        #result_sim.append(result_chunk_sim)

    q_sum_hh = jnp.transpose(jnp.vstack(result_hh))
    q_sum_hh = jnp.sum(q_sum_hh, axis = 1)
    #q_sum_sim = jnp.concatenate(result_sim, axis = 0)
    #q = jnp.exp(log_q)
    #q_sum = nansum_ignore_nan_inf_jitted(q)
    return q_sum_hh
get_q_sum_hh_jitted = jax.jit(get_q_sum_hh)

def get_q_sum_sim(p_l, q_l, fc_l):
    log_q = cf_w_jitted(p_l, q_l, fc_l)
    chunk_size = 100 
    num_columns = log_q.shape[1]
    #result_hh = []
    result_sim = []
    for start_col in range(0, num_columns, chunk_size):
        end_col = min(start_col + chunk_size, num_columns)
        data_chunk = log_q[:, start_col:end_col]
        result_chunk = jnp.exp(data_chunk)
        #result_chunk_hh = nansum_ignore_nan_inf_otheraxis_jitted(result_chunk)
        #result_hh.append(result_chunk_hh)
        result_chunk_sim = nansum_ignore_nan_inf_jitted(result_chunk)
        result_sim.append(result_chunk_sim)

    #q_sum_hh = jnp.transpose(jnp.vstack(result_hh))
    #q_sum_hh = jnp.sum(q_sum_hh, axis = 1)
    q_sum_sim = jnp.concatenate(result_sim, axis = 0)
    #q = jnp.exp(log_q)
    #q_sum = nansum_ignore_nan_inf_jitted(q)
    return q_sum_sim
get_q_sum_sim_jitted = jax.jit(get_q_sum_sim)

def get_r(r_mean, prem_id = None):
    if not prem_id:
        prem_id =  jnp.array(demand_2018_using_eta['prem_id'], dtype = jnp.int64)
    
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

def revenue_diff(r):
    return nansum_ignore_nan_inf_jitted(jnp.square((r - r0)) / r0)/hh_size
 
revenue_diff_jitted = jax.jit(revenue_diff)

q0_sum_mean = jnp.mean(q0_sum)

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
    
def revenue_non_exceeding(r):
    #condition = (r-r0)<0
    #return jnp.count_nonzero(condition)/len(q0_sum) - 0.95
    return nansum_ignore_nan_inf_jitted(r - r0)/hh_size
revenue_non_exceeding_jitted = jax.jit(revenue_non_exceeding)

def from_q_to_r(q_sum_hh, p_l, q_l, fc_l):
    q_mean = q_sum_hh / jnp.square(sim)
    r_mean = get_r_mean_jitted(q_mean, p_l, q_l, fc_l)
    r = get_r_jitted(r_mean)
    return r
from_q_to_r_jitted = jax.jit(from_q_to_r)

def param_to_pq (param):
    p_l = jnp.cumsum(jnp.array([param[0], param[1], param[2], param[3], param[4]]))
    q_l = jnp.cumsum(jnp.array([ 2, param[5], param[6], param[7]]))
    #q_l = jnp.cumsum(jnp.array([ param[5], param[6], param[7], param[8]]))
    #fc_l = jnp.array([param[9], param[10], param[11], param[12], param[13]])
    fc_l = jnp.cumsum(jnp.array([8.5, param[8], param[9], param[10], param[11]]))
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
    
def revenue_non_exceeding_constraint(param):
    p_l, q_l, fc_l = param_to_pq_jitted(param)
    q_sum_hh = get_q_sum_hh_jitted(p_l, q_l, fc_l)
    r = from_q_to_r_jitted(q_sum_hh, p_l, q_l, fc_l)
    return revenue_non_exceeding_jitted(r)

revenue_non_exceeding_constraint_jitted = jax.jit(revenue_non_exceeding_constraint)
#cons1 = {'type': 'ineq',
 #        'fun' : conservation_constraint_jitted}
#cons2 = {'type': 'ineq',
 #        'fun' : revenue_non_exceeding_constraint_jitted}
# Define the bounds as constraints
def bound1(x):
    return 3.09 - x[0]
bound1_jitted = jax.jit(bound1)

def bound2(x):
    return 4 - x[5]
bound2_jitted = jax.jit(bound2)

def bound3(x):
    return 5 - x[6]
bound3_jitted = jax.jit(bound3)

def bound4(x):
    return 9 - x[7]
bound4_jitted = jax.jit(bound4)

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

bounds = Bounds([0.01, 0.01, 0.01, 0.01, 0.01, 
                 #0.01,
                 0.01, 0.01, 0.01,
                 #0.01,
                 10.8-8.5, 16.5-10.8, 37-16.5, 
                 37-37
                 ], 
                [3.09, 10, 10, 10, 10,
                 #4,
                 5, 9, 11,
                 #10.8,
                 37, 37, 37, 
                 37
                 ])

#bounds = Bounds([0.01, 0.01, 0.01, 0.01, 0.01, 
#                 0.01, 
#                 0.01, 0.01, 0.01,
#                 0.01, 
#                0.01, 0.01, 0.01,
#                0.01
#                ], 
#                [jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, 
#                 jnp.inf, 
#                 jnp.inf, jnp.inf, jnp.inf, 
#                 jnp.inf, 
#                 jnp.inf, jnp.inf, jnp.inf, 
#                 jnp.inf
#                 ])

param0 = jnp.array([3.09, 5.01-3.09, 8.54-5.01, 12.9-8.54, 14.41-12.9, 
                    #2, 
                    6-2, 11-6, 20-11,
                    #8.5, 
                    10.8-8.5, 16.5-10.8, 37-16.5
                    , 37-37
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
solution = cobyqa.minimize(objective, param0, bounds = bounds, constraints=(constraint1, constraint2), options={'disp': True,
                                                                                                                'feasibility_tol': 0.0001, 
                                                                                                        #'radius_init':0.5, 
                                                                                                        'radius_final':0.0001
                                                                                                        })


#solution_0 = cobyqa.minimize(objective, param0, bounds = bounds, constraints=(constraint1, constraint2), options={'disp': True})

#np.savetxt('cf_risk_result_single.csv', solution, delimiter=',')
