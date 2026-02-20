import os
os.environ['JAX_ENABLE_X64'] = 'false'
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
from dataclasses import dataclass


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

eta_l_diff = jnp.array(demand_2018_using_eta['e_diff'])

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

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ModelParams:
    b1: jnp.ndarray
    b2: jnp.ndarray
    b4: jnp.ndarray
    b6: jnp.ndarray
    c_o: float
    c_alpha: float
    c_rho: float
    sigma_nu: float
    p_l0: jnp.ndarray
    q_l0: jnp.ndarray
    fc_l0: jnp.ndarray
    p_l0_CAP: jnp.ndarray

    def tree_flatten(self):
        children = (
            self.b1, self.b2, self.b4, self.b6,
            self.c_o, self.c_alpha, self.c_rho, self.sigma_nu,
            self.p_l0, self.q_l0, self.fc_l0, self.p_l0_CAP
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


params = ModelParams(
    b1=b1,
    b2=b2,
    b4=b4,
    b6=b6,
    c_o=c_o,
    c_alpha=c_alpha,
    c_rho=c_rho,
    sigma_nu=sigma_nu,
    p_l0=p_l0,
    q_l0=q_l0,
    fc_l0=fc_l0,
    p_l0_CAP=p_l0_CAP
)

 
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SegmentData:
    bathroom: jnp.ndarray
    bedroom: jnp.ndarray
    heavy_water_app: jnp.ndarray
    prem_id: jnp.ndarray
    bill_ym: jnp.ndarray
    CAP: jnp.ndarray
    income: jnp.ndarray
    p0: jnp.ndarray
    deflator: jnp.ndarray
    Z_current: jnp.ndarray
    segment_id: jnp.ndarray   # ← NEW

    def tree_flatten(self):
        children = (
            self.bathroom, self.bedroom, self.heavy_water_app,
            self.prem_id, self.bill_ym, self.CAP,
            self.income, self.p0, self.deflator,
            self.Z_current,
            self.segment_id
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def gen_nu_array(key, sigma, shape):
    nu = sigma * jax.random.normal(key, shape)
    return jnp.clip(nu, -7, 7)

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

rho = abs(jnp.dot(A_current_income, b6)
                    + c_rho)

demand_2018_using_new.loc[:, 'e_alpha'] = alpha
demand_2018_using_new.loc[:, 'e_rho'] = rho

CAP = jnp.array(demand_2018_using_new['CAP_HH'])

Z_current = Z_current_outdoor
roster_df = pd.read_csv('premise_segments_roster.csv')
demand_2018_using_new_small = pd.read_csv('demand_2018_using_new_small.csv')
panel_df = demand_2018_using_new_small.copy() # Ensure this DF exists
panel_with_labels = pd.merge(panel_df, roster_df[['prem_id', 'label']], on='prem_id', how='left')

def map_segment_4(label):
    if pd.isna(label):
        return 0  # 0 = missing (optional)
    
    first = label[0]
    
    if first == 'A':
        return 1
    if first == 'B':
        return 2
    if first == 'C':
        return 3
    if first == 'D':
        return 4
    
    return 0  # safety fallback

panel_with_labels['segment_type'] = (
    panel_with_labels['label']
        .apply(map_segment_4)
        .astype(int)
)

segment_type_array = jnp.array(
    panel_with_labels['segment_type'].values
)


segment_current = SegmentData(
    bathroom=jnp.array(demand_2018_using_new['bathroom']),
    bedroom=jnp.array(demand_2018_using_new['bedroom']),
    heavy_water_app=jnp.array(demand_2018_using_new['heavy_water_app']),
    prem_id=jnp.array(demand_2018_using_new['prem_id']),
    bill_ym=jnp.array(demand_2018_using_new['bill_ym']),
    CAP=jnp.array(demand_2018_using_new['CAP_HH']),
    income=jnp.array(demand_2018_using_new['income']),
    p0=jnp.array(demand_2018_using_new['previous_essential_usage_mp']),
    deflator=jnp.array(demand_2018_using_new['deflator']),
    Z_current=Z_current_outdoor,
    segment_id=segment_type_array
)


key = jax.random.PRNGKey(42)
N = segment_current.bathroom.shape[0]   # number of observations
sim = 100                               # your fixed MC size

nu_array = gen_nu_array(
    key,
    params.sigma_nu,
    (N, sim)
)



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

#### The plug in of NDVI is always current month
#### will return prev_NDVI after the process

def compute_new_ndvi(Z, Z_current, ndv, beta_prcp = beta_prcp):
    #print("Shape of Z:", Z.shape) 
    new_prcp = Z[:, 2]
    delta_prcp = new_prcp - Z_current[:, 2]
    return ndv + beta_prcp * delta_prcp

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

def calculate_log_w(params,
                    segment,
                    p_l,
                    q_l,
                    fc_l,
                    Z,
                    ndv):

    # unpack params
    p_l0 = params.p_l0
    p_l0_CAP = params.p_l0_CAP

    b1 = params.b1
    b2 = params.b2
    b4 = params.b4
    b6 = params.b6
    c_o = params.c_o
    c_alpha = params.c_alpha
    c_rho = params.c_rho

    # unpack segment data
    bathroom = segment.bathroom
    bedroom = segment.bedroom
    heavy_water_app = segment.heavy_water_app
    prem_id = segment.prem_id
    bill_ym = segment.bill_ym
    CAP = segment.CAP
    I = segment.income
    p0 = segment.p0
    de = segment.deflator
    Z_current = segment.Z_current

    # --- pricing adjustments ---
    p_l_CAP = p_l - p_l0 + p_l0_CAP

    q_kink_l = q_l

    p_plus1_l = jnp.append(p_l[1:5], jnp.array([jnp.nan]))
    d_end = jnp.insert(
        jnp.cumsum((p_plus1_l - 0.2 - (p_l - 0.2))[:4] * q_kink_l),
        0,
        0.0
    )

    p_plus1_l_CAP = jnp.append(p_l_CAP[1:5], jnp.array([jnp.nan]))
    d_end_CAP = jnp.insert(
        jnp.cumsum((p_plus1_l_CAP - 0.2 - (p_l_CAP - 0.2))[:4] * q_kink_l),
        0,
        0.0
    )

    # --- NDVI ---
    new_ndvi = compute_new_ndvi_jitted(Z, Z_current, ndv)
    prev_NDVI = update_prev_ndvi_jitted(new_ndvi, prem_id, bill_ym)

    A_o = jnp.column_stack((bathroom, prev_NDVI))
    A_i = jnp.column_stack((heavy_water_app, bedroom, prev_NDVI))
    A_p = jnp.column_stack((bedroom, prev_NDVI, Z[:, 0], Z[:, 2]))
    alpha = jnp.exp(jnp.dot(A_p, b4) + c_alpha)
    rho = jnp.abs(jnp.dot(A_i, b6) + c_rho)

    def compute_k(k):

        p_k = jnp.where(CAP == 1, p_l_CAP[k], p_l[k])
        d_k = jnp.where(CAP == 1,
                        -fc_l[k] + d_end_CAP[k],
                        -fc_l[k] + d_end[k])


        w = jnp.exp(
            jnp.dot(A_o, b1)
            + jnp.dot(Z, b2)
            - alpha * jnp.log(p_k) * de
            + rho * jnp.log(jnp.maximum(I + d_k * de, 1e-16))
            + c_o
        )

        return jnp.log(w)

    # Vectorize across tiers
    tiers = jnp.arange(5)

    log_w = jax.vmap(compute_k)(tiers)
    log_w = log_w.T


    return log_w


calculate_log_w_jitted = jax.jit(calculate_log_w)


def get_log_q_inner(log_w_k, q_l, sim):
    e = log_w_k[-1]

    # fixed structure
    log_w1 = log_w_k[0]
    log_w2 = log_w_k[1]
    log_w3 = log_w_k[2]
    log_w4 = log_w_k[3]
    log_w5 = log_w_k[4]

    # nu block is always from 5 to -1
    n = log_w_k[5:-1]
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

vmapped_log_q = jax.vmap(
    get_log_q_inner_jitted,
    in_axes=(0, None, None)
)

def get_log_w(params, segment,
              p_l, q_l, fc_l,
              Z, ndv,
              nu_array):

    log_w = calculate_log_w_jitted(
        params, segment,
        p_l, q_l, fc_l,
        Z, ndv
    )

    log_w = jnp.column_stack((log_w, nu_array))
    return log_w


get_log_w_jitted = jax.jit(get_log_w)


def get_log_q_sim(params, segment,
                  p_l, q_l, fc_l,
                  Z, ndv,
                  nu_array,
                  eta_l):

    log_w = get_log_w_jitted(
        params, segment,
        p_l, q_l, fc_l,
        Z, ndv,
        nu_array
    )

    log_w = jnp.column_stack((log_w, eta_l))

    # 🚀 REAL vectorization
    log_q_sim = vmapped_log_q(log_w, q_l, sim)

    return log_q_sim


get_log_q_sim_jitted = jax.jit(get_log_q_sim)

def cf_w(params, segment,
         p_l, q_l, fc_l,
         Z, ndv,
         nu_array,
         eta_l):

    log_q_sim = get_log_q_sim_jitted(
        params, segment,
        p_l, q_l, fc_l,
        Z, ndv,
        nu_array,
        eta_l
    )

    N = segment.bathroom.shape[0]
    sim = nu_array.shape[1]

    log_q_sim = log_q_sim.reshape(N, sim)


    return log_q_sim


cf_w_jitted = jax.jit(cf_w)

log_q0 = cf_w_jitted(
    params,
    segment_current,
    p_l0,
    q_l0,
    fc_l0,
    Z_current,
    NDVI,
    nu_array,
    eta_l
)


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

def get_r_mean(params, segment,
               q_mean,
               p_l, q_l, fc_l):

    p_l_CAP = p_l - params.p_l0 + params.p_l0_CAP

    def expenditure_func(w, p_l_local):

        bins = jnp.concatenate((jnp.array([0]), q_l, jnp.array([jnp.inf])))
        binned_data = jnp.digitize(w, bins)

        q_plus1_l = jnp.insert(q_l, 0, 0)
        q_diff_l = q_l - q_plus1_l[0:4]
        cumu_sum = jnp.cumsum(p_l_local[0:4] * q_diff_l)

        result = jnp.where(
            binned_data == 1,
            fc_l[0] + p_l_local[0] * w,
            fc_l[binned_data-1]
            + cumu_sum[binned_data-2]
            + p_l_local[binned_data-1]
            * (w - q_l[binned_data-2])
        )

        return result

    r_regular = expenditure_func(q_mean, p_l)
    r_cap     = expenditure_func(q_mean, p_l_CAP)

    r_mean = jnp.where(segment.CAP == 1, r_cap, r_regular)

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


def get_q_sum_hh(params, segment,
                 p_l, q_l, fc_l,
                 Z, ndv,
                 nu_array, eta_l):

    log_q = cf_w_jitted(
        params, segment,
        p_l, q_l, fc_l,
        Z, ndv,
        nu_array, eta_l
    )

    q = jnp.exp(log_q)

    # sum over simulations (axis=1)
    q_sum_hh = jnp.sum(q, axis=1)

    return q_sum_hh
get_q_sum_hh_jitted = jax.jit(get_q_sum_hh)

def get_q_sum_sim(params, segment,
                  p_l, q_l, fc_l,
                  Z, ndv,
                  nu_array, eta_l):

    log_q = cf_w_jitted(
        params, segment,
        p_l, q_l, fc_l,
        Z, ndv,
        nu_array, eta_l
    )

    q = jnp.exp(log_q)

    # sum over households (axis=0)
    q_sum_sim = jnp.sum(q, axis=0)

    return q_sum_sim
get_q_sum_sim_jitted = jax.jit(get_q_sum_sim)

######################
#### Revenue #####
########################

def from_q_to_r_mean(params, segment,
                     q_sum_hh,
                     p_l, q_l, fc_l):

    q_mean = q_sum_hh / sim

    r_mean = get_r_mean_jitted(
        params,
        segment,
        q_mean,
        p_l, q_l, fc_l
    )

    return r_mean

from_q_to_r_mean_jitted = jax.jit(from_q_to_r_mean)

def from_q_to_r(params, segment,
                q_sum_hh,
                p_l, q_l, fc_l):

    r_mean = from_q_to_r_mean_jitted(
        params,
        segment,
        q_sum_hh,
        p_l, q_l, fc_l
    )

    return r_mean

from_q_to_r_jitted = jax.jit(from_q_to_r)

def get_r(segment, r_mean):

    prem_id = segment.prem_id

    # sort by prem_id
    sort_idx = jnp.argsort(prem_id)
    sorted_ids = prem_id[sort_idx]
    sorted_r   = r_mean[sort_idx]

    # build segment ids
    unique_ids, segment_ids = jnp.unique(sorted_ids, return_inverse=True)

    # sum within household
    r_sum = lax.segment_sum(sorted_r, segment_ids)

    # counts per household
    counts = lax.segment_sum(jnp.ones_like(sorted_r), segment_ids)

    r_avg = r_sum / counts

    return r_avg


get_r_jitted = jax.jit(get_r)

q_sum_hh0 = get_q_sum_hh_jitted(
    params,
    segment_current,
    p_l0, q_l0, fc_l0,
    Z_current,
    NDVI,
    nu_array,
    eta_l
)

r0 = from_q_to_r_jitted(
    params,
    segment_current,
    q_sum_hh0,
    p_l0, q_l0, fc_l0
)



del A_current_indoor, demand_2018_using_new, demand_2018_using_eta, w_i

#Z_current_indoor
    

######################
#### Consumer Welfare #####
########################

def get_k(segment, q_sum_hh, q_l):

    q_mean = q_sum_hh / sim

    conditions_k = [
        (q_mean < q_l[0]),
        ((q_mean >= q_l[0]) & (q_mean < q_l[1])),
        ((q_mean >= q_l[1]) & (q_mean < q_l[2])),
        ((q_mean >= q_l[2]) & (q_mean < q_l[3])),
        (q_mean >= q_l[3]),
    ]

    choices = [0,1,2,3,4]

    return jnp.select(conditions_k, choices)

get_k_jitted = jax.jit(get_k)


def get_virtual_income(params, segment,
                       q_sum_hh,
                       p_l, q_l, fc_l):

    p_l_CAP = p_l - params.p_l0 + params.p_l0_CAP

    # build subsidy ladder
    p_plus1_l = jnp.append(p_l[1:5], jnp.array([jnp.nan]))
    d_end = jnp.insert(
        jnp.cumsum((p_plus1_l - 0.2 - (p_l - 0.2))[:4] * q_l),
        0,
        0.0
    )

    p_plus1_l_CAP = jnp.append(p_l_CAP[1:5], jnp.array([jnp.nan]))
    d_end_CAP = jnp.insert(
        jnp.cumsum((p_plus1_l_CAP - 0.2 - (p_l_CAP - 0.2))[:4] * q_l),
        0,
        0.0
    )

    k = get_k_jitted(segment, q_sum_hh, q_l)

    d_regular = -fc_l[k] + d_end[k]
    d_cap     = -fc_l[k] + d_end_CAP[k]

    d_k = jnp.where(segment.CAP == 1, d_cap, d_regular)

    virtual_income = jnp.maximum(d_k + segment.income, 1e-16)

    return virtual_income
get_virtual_income_jitted = jax.jit(get_virtual_income)

#alpha = jnp.exp(jnp.dot(A_current_price, b4)
 #                   + c_alpha)

def get_current_marginal_p(params, segment,
                           q_sum_hh,
                           p_l, q_l):

    p_l_CAP = p_l - params.p_l0 + params.p_l0_CAP

    k = get_k_jitted(segment, q_sum_hh, q_l)

    p_regular = p_l[k]
    p_cap     = p_l_CAP[k]

    return jnp.where(segment.CAP == 1, p_cap, p_regular)
get_current_marginal_p_jitted = jax.jit(get_current_marginal_p)

mean_nu_array =jnp.mean(nu_array, axis = 1)

demand_eta = demand_2018_using_new_season[['prem_id', 'bill_ym']]

demand_eta['eta'] = eta_l

demand_eta['mean_eta'] = demand_eta.groupby('prem_id')['eta'].transform('mean')

mean_eta_l = jnp.array(demand_eta['mean_eta'])

def get_expenditure_in_v_out(params, segment,
                             q_sum_hh,
                             p_l, q_l, fc_l,
                             Z, ndv,
                             nu_array, eta_l):

    # NDVI update
    new_ndvi = compute_new_ndvi_jitted(Z, segment.Z_current, ndv)
    prev_NDVI = update_prev_ndvi_jitted(
        new_ndvi,
        segment.prem_id,
        segment.bill_ym
    )

    # covariates
    A_o = jnp.column_stack((segment.bathroom, prev_NDVI))
    A_p = jnp.column_stack((segment.bedroom, prev_NDVI, Z[:,0], Z[:,2]))
    A_i = jnp.column_stack((segment.heavy_water_app,
                            segment.bedroom,
                            prev_NDVI))

    alpha = jnp.exp(jnp.dot(A_p, params.b4) + params.c_alpha)

    p = get_current_marginal_p(
        params, segment,
        q_sum_hh,
        p_l, q_l
    )

    mean_nu = jnp.mean(nu_array, axis=1)

    exp_factor = jnp.exp(
        jnp.dot(A_o, params.b1)
        + jnp.dot(Z, params.b2)
        + params.c_o
        + eta_l
        + mean_nu
    )

    tolerance = 1e-3
    alpha_minus_1 = alpha - 1.0

    price_component = jnp.where(
        jnp.abs(alpha_minus_1) < tolerance,
        jnp.log(p),
        jnp.power(p, -alpha_minus_1) / (-alpha_minus_1)
    )

    return exp_factor * price_component
get_expenditure_in_v_out_jitted = jax.jit(get_expenditure_in_v_out)

def get_v_out(params, segment,
              q_sum_hh,
              p_l, q_l, fc_l,
              Z, ndv,
              nu_array, eta_l):

    exp_v = get_expenditure_in_v_out(
        params, segment,
        q_sum_hh,
        p_l, q_l, fc_l,
        Z, ndv,
        nu_array, eta_l
    )

    Ik = get_virtual_income(
        params, segment,
        q_sum_hh,
        p_l, q_l, fc_l
    )

    new_ndvi = compute_new_ndvi_jitted(Z, segment.Z_current, ndv)
    prev_NDVI = update_prev_ndvi_jitted(
        new_ndvi,
        segment.prem_id,
        segment.bill_ym
    )

    A_i = jnp.column_stack((segment.heavy_water_app,
                        segment.bedroom,
                        prev_NDVI))

    rho = jnp.abs(jnp.dot(A_i, params.b6) + params.c_rho)

    tolerance = 1e-3
    rho_minus_1 = rho - 1.0

    second_term = jnp.where(
        jnp.abs(rho_minus_1) < tolerance,
        jnp.log(Ik),
        jnp.power(Ik, -rho_minus_1) / (-rho_minus_1)
    )

    return -exp_v + second_term
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

log_qhistory = cf_w_jitted(
    params,
    segment_current,
    p_l0, q_l0, fc_l0,
    Z_1417,
    NDVI,
    nu_array,
    eta_l
)

qhistory = jnp.exp(log_qhistory)

qhistory_sum =nansum_ignore_nan_inf_jitted(qhistory)
qhistory_mean = nanmean_ignore_nan_inf_jitted(qhistory)

q_sum_hhhistory = get_q_sum_hh_jitted(
    params,
    segment_current,
    p_l0,
    q_l0,
    fc_l0,
    Z_1417,
    NDVI,
    nu_array,
    eta_l
)

rhistory = from_q_to_r_jitted(params, segment_current,q_sum_hhhistory, p_l0, q_l0, fc_l0)

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
#Array(7186208., dtype=float32)
r_agg_0 = nansum_ignore_nan_inf_jitted(r0 )/12
#Array(28058074., dtype=float32)

#q_sum_hh_1417 = get_q_sum_hh_jitted(p_l0, q_l0, fc_l0, Z_1417)
#q0_filtered = q_sum_hh_current[q_sum_hh_current < 150000]
#q0_filtered = q_sum_hh_history[q_sum_hh_history < 150000]
q_agg_history = nansum_ignore_nan_inf_jitted(q_sum_hhhistory/100)/12
# Array(721182.7, dtype=float32)
q_agg_0 = nansum_ignore_nan_inf_jitted(q_sum_hh0/100)/12
#Array(2276729.8, dtype=float32)

cs_history = get_v_out_jitted(
    params,
    segment_current,
    q_sum_hhhistory,
    p_l0, q_l0, fc_l0,
    Z_1417,
    NDVI,
    nu_array,
    eta_l
)

#cs0_filtered = cs_0[(cs_0 > -0.5*1e9) ]
cs_agg_history = nansum_ignore_nan_inf_jitted(cs_history)/12
# Array(1.3572654e+09, dtype=float32)
cs_0 = get_v_out_jitted(
    params,
    segment_current,
    q_sum_hh0,
    p_l0, q_l0, fc_l0,
    Z_current,
    NDVI,
    nu_array,
    eta_l
)

cs_agg_0= nansum_ignore_nan_inf_jitted(cs_0)/12
#Array(1.3349313e+09, dtype=float32)

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

 
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class WelfareBaseline:
    mp_0: jnp.ndarray
    vi_0: jnp.ndarray
    cs_0: jnp.ndarray
    q0_sum_mean: float
    r0_sum: float

    def tree_flatten(self):
        children = (
            self.mp_0,
            self.vi_0,
            self.cs_0,
            self.q0_sum_mean,
            self.r0_sum
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def build_welfare_baseline(params, segment,
                           p_l0, q_l0, fc_l0,
                           Z, ndv,
                           nu_array, eta_l):

    # ---- Household quantity (for EV, CS, etc.)
    q_sum_hh0 = get_q_sum_hh_jitted(
        params, segment,
        p_l0, q_l0, fc_l0,
        Z, ndv,
        nu_array, eta_l
    )

    # ---- Simulation aggregate quantity (for conservation)
    q_sum_sim0 = get_q_sum_sim_jitted(
        params, segment,
        p_l0, q_l0, fc_l0,
        Z, ndv,
        nu_array, eta_l
    )

    q0_sum_mean = jnp.mean(q_sum_sim0)

    # ---- Marginal price
    k0 = get_k_jitted(segment, q_sum_hh0, q_l0)

    p_l0_CAP = p_l0 - params.p_l0 + params.p_l0_CAP

    mp_0 = jnp.where(
        segment.CAP == 1,
        p_l0_CAP[k0],
        p_l0[k0]
    )

    # ---- Virtual income
    vi_0 = get_virtual_income(
        params, segment,
        q_sum_hh0,
        p_l0, q_l0, fc_l0
    )

    # ---- Baseline CS
    cs_0 = get_v_out(
        params, segment,
        q_sum_hh0,
        p_l0, q_l0, fc_l0,
        Z, ndv,
        nu_array, eta_l
    )

    # ---- Baseline revenue
    r0 = from_q_to_r_jitted(
        params,
        segment,
        q_sum_hh0,
        p_l0, q_l0, fc_l0
    )

    r0_sum = nansum_ignore_nan_inf_jitted(r0)

    return WelfareBaseline(mp_0, vi_0, cs_0, q0_sum_mean, r0_sum)


baseline = build_welfare_baseline(
    params,
    segment_current,
    p_l0, q_l0, fc_l0,
    Z_current,
    NDVI,
    nu_array,
    eta_l
)

def find_q_sum_hh_close_to_ql(segment,
                              all_q_sum_hh,
                              q_l,
                              tolerance=1e-3):

    q_hh = all_q_sum_hh / sim

    diffs = jnp.abs(q_hh[:, None] - q_l[None, :])

    is_close = jnp.any(diffs < tolerance, axis=1)

    nearest_idx = jnp.argmin(diffs, axis=1)
    nearest_val = q_l[nearest_idx]

    return is_close, nearest_idx, nearest_val


@jax.jit
def solve_for_pbar_vectorized(tk, A, alpha, rho, pk, vi0):

    def scalar_solver(tk_, A_, alpha_, rho_, pk_, vi0_, eps=1e-6):

        log_tk = jnp.log(tk_)

        def f(pbar):

            near_alpha_1 = jnp.abs(1.0 - alpha_) < eps
            near_rho_1   = jnp.abs(1.0 - rho_)   < eps

            # Power difference
            p_diff = jnp.where(
                near_alpha_1,
                jnp.log(pbar) - jnp.log(pk_),
                pbar**(1 - alpha_) - pk_**(1 - alpha_)
            )

            alpha_term = jnp.where(
                near_alpha_1,
                1.0,
                (1 - alpha_) / (1 - rho_)
            )

            # Correct limit: (I+d0)^(1-rho) → 1
            Id_term = jnp.where(
                near_rho_1,
                1.0,
                vi0_**(1 - rho_)
            )

            inner = jnp.exp(A_) * alpha_term * p_diff + Id_term
            inner = jnp.maximum(inner, 1e-12)

            f_val = A_ - alpha_ * jnp.log(pbar)

            f_val += jnp.where(
                near_rho_1,
                jnp.log(inner),   # correct limit
                (rho_ / (1 - rho_)) * jnp.log(inner)
            )

            return f_val - log_tk

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

        _, _, root, _ = jax.lax.while_loop(
            cond, body, (lower, upper, 0.0, 0)
        )

        return root

    return jax.vmap(scalar_solver)(tk, A, alpha, rho, pk, vi0)

def get_e_new_v_p0(params, segment,
                   baseline,
                   q_sum_hh,
                   v_out,
                   p_l, q_l, fc_l,
                   Z, ndv,
                   nu_array, eta_l):

    # -------------------------------------------------
    # 1️⃣ Detect kink households
    # -------------------------------------------------
    is_close, _, nearest_val = find_q_sum_hh_close_to_ql(
        segment, q_sum_hh, q_l
    )

    # -------------------------------------------------
    # 2️⃣ Recompute covariates
    # -------------------------------------------------
    new_ndvi = compute_new_ndvi_jitted(Z, segment.Z_current, ndv)

    prev_NDVI = update_prev_ndvi_jitted(
        new_ndvi,
        segment.prem_id,
        segment.bill_ym
    )

    A_o = jnp.column_stack((segment.bathroom, prev_NDVI))
    A_p = jnp.column_stack((segment.bedroom, prev_NDVI, Z[:,0], Z[:,2]))
    A_i = jnp.column_stack((segment.heavy_water_app,
                            segment.bedroom,
                            prev_NDVI))

    alpha = jnp.exp(jnp.dot(A_p, params.b4) + params.c_alpha)
    rho   = jnp.abs(jnp.dot(A_i, params.b6) + params.c_rho)

    mean_nu = jnp.mean(nu_array, axis=1)

    A_term = jnp.exp(
        jnp.dot(A_o, params.b1)
        + jnp.dot(Z, params.b2)
        + params.c_o
        + eta_l
        + mean_nu
    )

    mp0 = baseline.mp_0

    # -------------------------------------------------
    # 3️⃣ Solve for p_bar only where kink occurs
    # -------------------------------------------------
    tk_safe = jnp.where(is_close, nearest_val, mp0)

    p_bar_all = solve_for_pbar_vectorized(
        tk_safe,
        A_term,
        alpha,
        rho,
        mp0,
        baseline.vi_0
    )

    p_bar = jnp.where(is_close, p_bar_all, mp0)

    # -------------------------------------------------
    # 4️⃣ Expenditure inversion
    # -------------------------------------------------
    tolerance = 1e-3
    alpha_minus_1 = alpha - 1.0

    mp_term = jnp.where(
        jnp.abs(alpha_minus_1) < tolerance,
        jnp.log(p_bar),
        jnp.power(p_bar, -alpha_minus_1) / (-alpha_minus_1)
    )

    base = jnp.maximum(
        (1 - rho) * (v_out + A_term * mp_term),
        1e-16
    )

    e_unscaled = jnp.where(
        jnp.abs(1 - rho) < tolerance,
        0.0,
        jnp.power(base, 1.0 / (1 - rho))
    )

    # -------------------------------------------------
    # 5️⃣ Discrete kink correction
    # -------------------------------------------------
    additional = jnp.where(
        is_close,
        (p_bar - mp0) * nearest_val,
        0.0
    )

    e_corrected = e_unscaled - additional

    return e_corrected



def get_ev(params, segment,
           baseline,
           q_sum_hh,
           v_out,
           p_l, q_l, fc_l,
           Z, ndv,
           nu_array, eta_l):

    e = get_e_new_v_p0(
        params, segment,
        baseline,
        q_sum_hh, v_out,
        p_l, q_l, fc_l,
        Z, ndv,
        nu_array, eta_l
    )

    unscaled_ev = e - baseline.vi_0

    ev = unscaled_ev * segment.deflator

    return ev

########################
#### Revenue Conditions #####
########################
rhistory_sum_filtered =  nansum_ignore_nan_inf_jitted(rhistory)
r0_sum_filtered =  nansum_ignore_nan_inf_jitted(r0 )

rhistory_soft_constraint = 0.8*rhistory_sum_filtered

r0_soft_constraint = 0.8*r0_sum_filtered

log_r0_mean_filtered = jnp.log(r0_sum_filtered/12)

r0_mean_filtered = r0_sum_filtered/len_transactions

    
def revenue_compare(r, r0_benchmark):
    ### Here r is for each transaction, compare to r0 as the avg cost for each transaction
    return r - r0_benchmark

revenue_compare_jitted = jax.jit(revenue_compare)

########################
#### Conservation Conditions #####
########################

def cf_w_ci(segment,
            baseline,
            q_sum_sim):

    condition = (baseline.q0_sum_mean - q_sum_sim) > 0

    return jnp.count_nonzero(condition)


def conservation_condition(segment, baseline, q_sum_sim):
    q_mean = jnp.mean(q_sum_sim)
    return baseline.q0_sum_mean - q_mean


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

prev_weight = jnp.divide(1, I)

def create_median_normalized_weights(income_array):
    """
    Calculates equity weights for a social welfare function, normalized by the median.

    This method is preferred for highly skewed income distributions as it is robust
    to outliers. A household whose inverse income is the median will receive a
    weight of 1. Lower-income households get a weight > 1, and higher-income
    households get a weight < 1.

    Args:
        income_array: A JAX numpy array of household incomes.

    Returns:
        A JAX numpy array of median-normalized weights.
    """
    # To avoid division by zero or negative incomes, add a small epsilon
    # or ensure data is clean beforehand.
    safe_incomes = jnp.maximum(income_array, 1e-6)

    # 1. Calculate the inverse of the incomes
    inverse_incomes = 1.0 / safe_incomes

    # 2. Find the median of the inverse incomes
    median_of_inverse_incomes = jnp.median(inverse_incomes)

    # 3. Create the normalized weights
    normalized_weights = inverse_incomes / median_of_inverse_incomes

    return normalized_weights

equity_weights = create_median_normalized_weights(segment_current.income)
 
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OptimizationContext:
    params: ModelParams
    segment: SegmentData
    baseline: WelfareBaseline
    equity_weights: jnp.ndarray
    lambda_penalty: float

    def tree_flatten(self):
        children = (
            self.params,
            self.segment,
            self.baseline,
            self.equity_weights,
            self.lambda_penalty
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


context = OptimizationContext(
    params=params,
    segment=segment_current,
    baseline=baseline,
    equity_weights=equity_weights,
    lambda_penalty=lambda_
)

def param_to_pq0 (param):
    p_l = jnp.cumsum(jnp.array([param[0], param[1], param[2], param[3], param[4]]))

    q_l = jnp.cumsum(jnp.array([ param[5], param[6], param[7], param[8]]))

    fc_l = jnp.cumsum(jnp.array([param[9], param[10], param[11], param[12], param[13]]))

    return p_l, q_l, fc_l

param_to_pq0_jitted = jax.jit(param_to_pq0)

###############################################################
######### Create Segments #####################
###############################################################


def get_welfare_and_revenue(context,
                            p_l, q_l, fc_l,
                            Z, ndv,
                            nu_array, eta_l):

    params   = context.params
    segment  = context.segment
    weights  = context.equity_weights

    # Quantity
    q_sum_hh = get_q_sum_hh_jitted(
        params, segment,
        p_l, q_l, fc_l,
        Z, ndv,
        nu_array, eta_l
    )

    # Revenue
    r = from_q_to_r_jitted(
        params,
        segment,
        q_sum_hh,
        p_l, q_l, fc_l
    )

    r_sum = nansum_ignore_nan_inf_jitted(r)

    # Consumer surplus
    cs = get_v_out(
        params, segment,
        q_sum_hh,
        p_l, q_l, fc_l,
        Z, ndv,
        nu_array, eta_l
    )

    # Equivalent variation
    ev = get_ev(
        params, segment,
        context.baseline,
        q_sum_hh, cs,
        p_l, q_l, fc_l,
        Z, ndv,
        nu_array, eta_l
    )

    welfare = nansum_ignore_nan_inf_jitted(ev * weights)

    return welfare, r_sum
get_welfare_and_revenue_jitted = jax.jit(get_welfare_and_revenue)

param0 = jnp.array([3.09, 5.01-3.09, 8.54-5.01, 12.9-8.54, 14.41-12.9, 
                   # 2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    10.8-8.5, 16.5-10.8, 37-16.5
                    , 37-37
                    ])

segment_id = segment_current.segment_id

mask_A = (segment_id == 1)
mask_B = (segment_id == 2)
mask_C = (segment_id == 3)
mask_D = (segment_id == 4)

###############################################################
######### Just seperate to 2 segments #####################
###############################################################

elastic_mask = jnp.isin(segment_current.segment_id, jnp.array([1,2]))
inelastic_mask = jnp.isin(segment_current.segment_id, jnp.array([3,4]))

print("Elastic mean income:", segment_current.income[elastic_mask].mean())
print("Inelastic mean income:", segment_current.income[inelastic_mask].mean())

def subset_segment(segment, mask):
    return SegmentData(
        bathroom=segment.bathroom[mask],
        bedroom=segment.bedroom[mask],
        heavy_water_app=segment.heavy_water_app[mask],
        prem_id=segment.prem_id[mask],
        bill_ym=segment.bill_ym[mask],
        CAP=segment.CAP[mask],
        income=segment.income[mask],
        p0=segment.p0[mask],
        deflator=segment.deflator[mask],
        Z_current=segment.Z_current[mask],
        segment_id=segment.segment_id[mask]
    )

segment_e = subset_segment(segment_current, elastic_mask)
segment_i = subset_segment(segment_current, inelastic_mask)

weights_e = create_median_normalized_weights(segment_e.income)
weights_i = create_median_normalized_weights(segment_i.income)

Z_e = Z_current[elastic_mask]
Z_i = Z_current[inelastic_mask]

ndv_e = NDVI[elastic_mask]
ndv_i = NDVI[inelastic_mask]

nu_e = nu_array[elastic_mask, :]
nu_i = nu_array[inelastic_mask, :]

eta_e = eta_l[elastic_mask]
eta_i = eta_l[inelastic_mask]

baseline_full = build_welfare_baseline(
    params,
    segment_current,
    p_l0, q_l0, fc_l0,
    Z_current,
    NDVI,
    nu_array,
    eta_l
)

baseline_e = WelfareBaseline(
    mp_0 = baseline_full.mp_0[elastic_mask],
    vi_0 = baseline_full.vi_0[elastic_mask],
    cs_0 = baseline_full.cs_0[elastic_mask],
    q0_sum_mean = None,   # scalar — same for all
    r0_sum = None              # scalar — same for all
)

baseline_i = WelfareBaseline(
    mp_0 = baseline_full.mp_0[inelastic_mask],
    vi_0 = baseline_full.vi_0[inelastic_mask],
    cs_0 = baseline_full.cs_0[inelastic_mask],
    q0_sum_mean = None,
    r0_sum = None
)


context_e = OptimizationContext(
    params=params,
    segment=segment_e,
    baseline=baseline_e,
    equity_weights=weights_e,
    lambda_penalty=lambda_
)

context_i = OptimizationContext(
    params=params,
    segment=segment_i,
    baseline=baseline_i,
    equity_weights=weights_i,
    lambda_penalty=lambda_
)

def param_to_pq0_PD(param):

    p_base = jnp.cumsum(param[0:5])

    delta = jnp.maximum(param[5:10], 0.0)
    p_inelastic = p_base + delta

    q_l = jnp.cumsum(param[10:14])

    fc_e = jnp.cumsum(param[14:19])
    fc_i = jnp.cumsum(param[19:24])

    return p_base, p_inelastic, q_l, fc_e, fc_i

param_to_pq0_PD_jitted = jax.jit(param_to_pq0_PD)

def objective_PD(param,
                 context_e,
                 context_i,
                 Z_e, Z_i,
                 ndv_e, ndv_i,
                 nu_e, nu_i,
                 eta_e, eta_i):

    param = jnp.maximum(param, 0.01)

    p_base, p_inelastic, q_l, fc_e, fc_i = param_to_pq0_PD_jitted(param)

    # ----- Elastic segment -----
    welfare_e, revenue_e = get_welfare_and_revenue(
        context_e,
        p_base, q_l, fc_e,
        Z_e, ndv_e,
        nu_e, eta_e
    )

    welfare_i, revenue_i = get_welfare_and_revenue(
        context_i,
        p_inelastic, q_l, fc_i,
        Z_i, ndv_i,
        nu_i, eta_i
    )


    # Aggregate
    total_welfare = welfare_e + welfare_i
    total_revenue = revenue_e + revenue_i

    #revenue_penalty = context_e.lambda_penalty * loss_function_quadratic(total_revenue - baseline_full.r0_sum)


    #return -(total_welfare - revenue_penalty)
    return -(total_welfare)
objective_PD_jitted = jax.jit(objective_PD)

def revenue_constraint_PD(param,
                          context_e,
                          context_i,
                          Z_e, Z_i,
                          ndv_e, ndv_i,
                          nu_e, nu_i,
                          eta_e, eta_i):

    param = jnp.maximum(param, 0.01)

    p_base, p_inelastic, q_l, fc_e, fc_i = param_to_pq0_PD_jitted(param)

    welfare_e, revenue_e = get_welfare_and_revenue(
        context_e,
        p_base, q_l, fc_e,
        Z_e, ndv_e,
        nu_e, eta_e
    )

    welfare_i, revenue_i = get_welfare_and_revenue(
        context_i,
        p_inelastic, q_l, fc_i,
        Z_i, ndv_i,
        nu_i, eta_i
    )

    total_revenue = revenue_e + revenue_i

    return total_revenue - baseline_full.r0_sum

def conservation_constraint_PD(param,
                               context_e,
                               context_i,
                               Z_e, Z_i,
                               ndv_e, ndv_i,
                               nu_e, nu_i,
                               eta_e, eta_i):

    param = jnp.maximum(param, 0.01)

    # --- unpack PD parameters ---
    p_base, p_inelastic, q_l, fc_e, fc_i = param_to_pq0_PD_jitted(param)

    # ----- Elastic segment -----
    q_sim_e = get_q_sum_sim_jitted(
        context_e.params,
        context_e.segment,
        p_base, q_l, fc_e,
        Z_e, ndv_e,
        nu_e, eta_e
    )

    # ----- Inelastic segment -----
    q_sim_i = get_q_sum_sim_jitted(
        context_i.params,
        context_i.segment,
        p_inelastic, q_l, fc_i,
        Z_i, ndv_i,
        nu_i, eta_i
    )

    # Aggregate total usage
    q_total_sim = q_sim_e + q_sim_i

    # Apply conservation rule on TOTAL
    return baseline_full.q0_sum_mean - jnp.mean(q_total_sim)

conservation_constraint_PD_jitted = jax.jit(conservation_constraint_PD)

constraint_PD = NonlinearConstraint(
    lambda x: conservation_constraint_PD(
        x,
        context_e,
        context_i,
        Z_e, Z_i,
        ndv_e, ndv_i,
        nu_e, nu_i,
        eta_e, eta_i
    ),
    0.0,
    jnp.inf
)

revenue_constraint = NonlinearConstraint(
    lambda x: revenue_constraint_PD(
        x,
        context_e, context_i,
        Z_e, Z_i,
        ndv_e, ndv_i,
        nu_e, nu_i,
        eta_e, eta_i
    ),
    0.0,
    jnp.inf
)


################ The optimal FP results from changing weather ###########
fp1 = jnp.array([8.304747, 15.349133, 22.364378, 29.401863, 36.494286])
fp2 = jnp.array([8.525296, 15.7042, 22.76314, 29.841204, 36.938675])

fp_avg = (fp1 + fp2) / 2
print(fp_avg)

fp_avg_increments = jnp.diff(
    jnp.insert(fp_avg, 0, 0.0)
)

fp_lower = fp_avg_increments

param_0 = jnp.array([3.09, 5.01-3.09, 8.54-5.01, 12.9-8.54, 14.41-12.9, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    10.8-8.5, 16.5-10.8, 37-16.5
                    , 37-37
                    ])

param_universal_opt = jnp.array([3, 5, 5, 5, 5, 
                    2, 
                    6-2, 11-6, 20-11,
                    8.5, 
                    7.125, 7.125,7.125,7.125
                    ])

param0_PD = jnp.concatenate([
    param_universal_opt[0:5],    # base marginal increments
    jnp.zeros(5),                # wedges
    param_universal_opt[5:9],    # kink increments
    param_universal_opt[9:14],    # fixed increments
    param_universal_opt[9:14]    # fixed increments
])


lower_bounds = (
    [0.01]*5 +      # base
    [0.01]*5 +      # wedges
    [0.01]*4 +      # kinks
    [0.01]*5 +      # fc_e
    [0.01]*5        # fc_i
)

upper_bounds = (
    [30]*5 +
    [15]*5 +
    [50]*4 +
    [100]*5 +
    [100]*5
)


bounds_PD = Bounds(lower_bounds, upper_bounds)
#################################
#### Multi Start####
#################################
def generate_feasible_start(key,
                            baseline_full,
                            context_e, context_i,
                            Z_e, Z_i,
                            ndv_e, ndv_i,
                            nu_e, nu_i,
                            eta_e, eta_i,
                            lower_bounds,
                            upper_bounds):

    dim = lower_bounds.shape[0]

    u = jax.random.uniform(key, shape=(dim,))
    param = lower_bounds + u * (upper_bounds - lower_bounds)

    param = jnp.maximum(param, 0.01)

    return param


def run_multistart(n_starts,
                   baseline_full,
                   context_e, context_i,
                   Z_e, Z_i,
                   ndv_e, ndv_i,
                   nu_e, nu_i,
                   eta_e, eta_i,
                   bounds_PD):

    best_obj = jnp.inf
    best_param = None

    key = jax.random.PRNGKey(0)

    for i in range(n_starts):

        key, subkey = jax.random.split(key)

        param0 = generate_feasible_start(
            subkey,
            baseline_full,
            context_e, context_i,
            Z_e, Z_i,
            ndv_e, ndv_i,
            nu_e, nu_i,
            eta_e, eta_i,
            jnp.array(bounds_PD.lb),
            jnp.array(bounds_PD.ub)
        )

        sol = cobyqa.minimize(
            lambda x: objective_PD_jitted(
                x,
                context_e, context_i,
                Z_e, Z_i,
                ndv_e, ndv_i,
                nu_e, nu_i,
                eta_e, eta_i
            ),
            np.array(param0),
            bounds=bounds_PD,
            constraints=(constraint_PD,revenue_constraint),
            options={'disp': False,
                     'feasibility_tol': 1e-6,
                     'radius_init': 1,
                     'radius_final': 0.1}
        )

        obj_val = sol.fun

        if obj_val < best_obj:
            best_obj = obj_val
            best_param = sol.x

        print(f"Start {i+1}/{n_starts}, objective: {obj_val}")

    return jnp.array(best_param)

best_param = run_multistart(
    n_starts=10,
    baseline_full=baseline_full,
    context_e=context_e,
    context_i=context_i,
    Z_e=Z_e, Z_i=Z_i,
    ndv_e=ndv_e, ndv_i=ndv_i,
    nu_e=nu_e, nu_i=nu_i,
    eta_e=eta_e, eta_i=eta_i,
    bounds_PD=bounds_PD
)

#################################
#### CHeck Initial Constriant####
#################################

best_param = param0_PD
val = conservation_constraint_PD(
    best_param,
    context_e,
    context_i,
    Z_e, Z_i,
    ndv_e, ndv_i,
    nu_e, nu_i,
    eta_e, eta_i
)

print("Initial conservation value:", val)

val = revenue_constraint_PD(
    best_param,
    context_e,
    context_i,
    Z_e, Z_i,
    ndv_e, ndv_i,
    nu_e, nu_i,
    eta_e, eta_i
)


print("Initial revenue value:", val)
#################################
#### Solving####
#################################
solution_PD = cobyqa.minimize(
    lambda x: objective_PD_jitted(
        x,
        context_e, context_i,
        Z_e, Z_i,
        ndv_e, ndv_i,
        nu_e, nu_i,
        eta_e, eta_i
    ),
    np.array(best_param),
    bounds=bounds_PD,
    constraints=(constraint_PD,revenue_constraint),
    options={
        'disp': True,
        'feasibility_tol': 1e-8,
        'radius_init': 0.5,
        'radius_final': 0.01
    }
)

param_star = jnp.array(solution_PD.x)

print("Solution success:", solution_PD.success)
print("Final conservation value:",
      conservation_constraint_PD(
          param_star,
          context_e, context_i,
          Z_e, Z_i,
          ndv_e, ndv_i,
          nu_e, nu_i,
          eta_e, eta_i
      )
)

p_base, p_inelastic, q_l, fc_e, fc_i = param_to_pq0_PD_jitted(param_star)

wedges = p_inelastic - p_base

print("Base marginal prices:", p_base)
print("Inelastic marginal prices:", p_inelastic)
print("Price wedges:", wedges)
print("Kinks:", q_l)
print("Fixed payments_elastic:", fc_e)
print("Fixed payments_inelastic:", fc_i)

###### param_uni is the optimal pricing result from the status quo weather with initial value of param_0_med

param_uni = jnp.array([0.53939277, 3.040236 - 0.53939277, 6.952795-3.040236, 11.379787 - 6.952795, 16.221025 - 11.379787,
                    2.6897972, 7.038082 - 2.6897972, 12.079354 - 7.038082, 21.25452 - 12.079354,
                    8.304747, 15.349133 - 8.304747, 22.364378 - 15.349133, 29.401863 - 22.364378 , 36.494286 -29.401863
                    ])

p_uni, q_uni, fc_uni = param_to_pq0_jitted(param_0 )

welfare_uniform, revenue_uniform = get_welfare_and_revenue_jitted(
    context,
    p_uni, q_uni, fc_uni,
    Z_current,
    NDVI,
    nu_array,
    eta_l
)

welfare_e, revenue_e = get_welfare_and_revenue_jitted(
    context_e,
    p_base, q_l, fc_e,
    Z_e, ndv_e,
    nu_e, eta_e
)

welfare_i, revenue_i = get_welfare_and_revenue_jitted(
    context_i,
    p_inelastic, q_l, fc_i,
    Z_i, ndv_i,
    nu_i, eta_i
)

welfare_PD = welfare_e + welfare_i
revenue_PD = revenue_e + revenue_i


print("Uniform welfare:", float(welfare_uniform))
print("PD welfare:", float(welfare_PD))
print("Welfare gain:", float(welfare_PD - welfare_uniform))

print("Uniform revenue:", float(revenue_uniform))
print("PD revenue:", float(revenue_PD))


#################################
#### Export Results ####
#################################
p_base, p_inelastic, q_l, fc_e, fc_i = param_to_pq0_PD_jitted(param_star)

output_dir = "NDVI_categorization/results"
os.makedirs(output_dir, exist_ok=True)

pd.DataFrame(p_base).to_csv(
    f"{output_dir}/PD_price_base_revenue_cons.csv",
    index=False
)

pd.DataFrame(p_inelastic).to_csv(
    f"{output_dir}/PD_price_inelastic_revenue_cons.csv",
    index=False
)

pd.DataFrame(q_l).to_csv(
    f"{output_dir}/PD_kinks_revenue_cons.csv",
    index=False
)

pd.DataFrame(fc_e).to_csv(
    f"{output_dir}/PD_fixed_payments_elastic_revenue_cons.csv",
    index=False
)

pd.DataFrame(fc_i).to_csv(
    f"{output_dir}/PD_fixed_payments_inelastic_revenue_cons.csv",
    index=False
)

q_e = get_q_sum_hh_jitted(
    context_e.params, context_e.segment,
    p_base, q_l, fc_e,
    Z_e, ndv_e,
    nu_e, eta_e
) / sim

r_e = from_q_to_r_jitted(
    context_e.params,
    context_e.segment,
    q_e,
    p_base, q_l, fc_e
)

cs_e = get_v_out(
    context_e.params, context_e.segment,
    q_e,
    p_base, q_l, fc_e,
    Z_e, ndv_e,
    nu_e, eta_e
)

ev_e = get_ev(
    context_e.params, context_e.segment,
    context_e.baseline,
    q_e, cs_e,
    p_base, q_l, fc_e,
    Z_e, ndv_e,
    nu_e, eta_e
)


q_i = get_q_sum_hh_jitted(
    context_i.params, context_i.segment,
    p_inelastic, q_l, fc_i,
    Z_i, ndv_i,
    nu_i, eta_i
)/sim

r_i = from_q_to_r_jitted(
    context_i.params,
    context_i.segment,
    q_i,
    p_inelastic, q_l, fc_i
)

cs_i = get_v_out(
    context_i.params, context_i.segment,
    q_i,
    p_inelastic, q_l, fc_i,
    Z_i, ndv_i,
    nu_i, eta_i
)
ev_i = get_ev(
    context_i.params, context_i.segment,
    context_i.baseline,
    q_i, cs_i,
    p_inelastic, q_l, fc_i,
    Z_i, ndv_i,
    nu_i, eta_i
)


q_full = jnp.zeros_like(segment_current.income)

q_full = q_full.at[elastic_mask].set(q_e)
q_full = q_full.at[inelastic_mask].set(q_i)

r_full = jnp.zeros_like(segment_current.income)

r_full = r_full.at[elastic_mask].set(r_e)
r_full = r_full.at[inelastic_mask].set(r_i)

cs_full = jnp.zeros_like(segment_current.income)

cs_full = cs_full.at[elastic_mask].set(cs_e)
cs_full = cs_full.at[inelastic_mask].set(cs_i)

ev_full = jnp.zeros_like(segment_current.income)

ev_full = ev_full.at[elastic_mask].set(ev_e)
ev_full = ev_full.at[inelastic_mask].set(ev_i)


hh_df = pd.DataFrame({
    "prem_id": np.array(segment_current.prem_id),
    "segment_id": np.array(segment_current.segment_id),
    "q": np.array(q_full),
    "r": np.array(r_full),
    "cs": np.array(cs_full),
    "ev": np.array(ev_full),
})

hh_df.to_csv(
    f"{output_dir}/PD_hh_level_results_revenue_cons.csv",
    index=False
)

welfare_total = (
    nansum_ignore_nan_inf_jitted(ev_e * weights_e) +
    nansum_ignore_nan_inf_jitted(ev_i * weights_i)
)


revenue_total = (
    nansum_ignore_nan_inf_jitted(r_e) +
    nansum_ignore_nan_inf_jitted(r_i)
)



summary_df = pd.DataFrame({
    "total_welfare": [float(welfare_total)],
    "total_revenue": [float(revenue_total)],
})

summary_df.to_csv(
    f"{output_dir}/PD_aggregate_summary_revenue_cons.csv",
    index=False
)

###############################################################
######### Naive way of 3DPD #####################
###############################################################

median_ndvi = jnp.median(NDVI)

low_mask  = NDVI <= median_ndvi
high_mask = NDVI >  median_ndvi

segment_low  = subset_segment(segment_current, low_mask)
segment_high = subset_segment(segment_current, high_mask)

weights_low = create_median_normalized_weights(segment_low.income)
weights_high = create_median_normalized_weights(segment_high.income)

Z_low  = Z_current[low_mask]
Z_high = Z_current[high_mask]

ndv_low  = NDVI[low_mask]
ndv_high = NDVI[high_mask]

nu_low  = nu_array[low_mask, :]
nu_high = nu_array[high_mask, :]

eta_low  = eta_l[low_mask]
eta_high = eta_l[high_mask]

baseline_low = WelfareBaseline(
    mp_0 = baseline_full.mp_0[low_mask],
    vi_0 = baseline_full.vi_0[low_mask],
    cs_0 = baseline_full.cs_0[low_mask],
    q0_sum_mean = None,   # scalar — same for all
    r0_sum = None              # scalar — same for all
)

baseline_high = WelfareBaseline(
    mp_0 = baseline_full.mp_0[high_mask],
    vi_0 = baseline_full.vi_0[high_mask],
    cs_0 = baseline_full.cs_0[high_mask],
    q0_sum_mean = None,
    r0_sum = None
)


context_low = OptimizationContext(
    params=params,
    segment=segment_low,
    baseline=baseline_low,
    equity_weights=weights_low,
    lambda_penalty=lambda_
)

context_high = OptimizationContext(
    params=params,
    segment=segment_high,
    baseline=baseline_high,
    equity_weights=weights_high,
    lambda_penalty=lambda_
)

best_param = param0_PD
val = conservation_constraint_PD(
    best_param,
    context_low,
    context_high,
    Z_low, Z_high,
    ndv_low, ndv_high,
    nu_low, nu_high,
    eta_low, eta_high
)

print("Initial conservation value:", val)

val = revenue_constraint_PD(
    best_param,
    context_low,
    context_high,
    Z_low, Z_high,
    ndv_low, ndv_high,
    nu_low, nu_high,
    eta_low, eta_high
)


print("Initial revenue value:", val)
#################################
#### Solving####
#################################
solution_PD = cobyqa.minimize(
    lambda x: objective_PD_jitted(
        x,
        context_low, context_high,
        Z_low, Z_high,
        ndv_low, ndv_high,
        nu_low, nu_high,
        eta_low, eta_high
    ),
    np.array(best_param),
    bounds=bounds_PD,
    constraints=(constraint_PD,revenue_constraint),
    options={
        'disp': True,
        'feasibility_tol': 1e-8,
        'radius_init': 0.5,
        'radius_final': 0.01
    }
)

param_star = jnp.array(solution_PD.x)

print("Solution success:", solution_PD.success)
print("Final conservation value:",
      conservation_constraint_PD(
          param_star,
          context_low, context_high,
          Z_low, Z_high,
          ndv_low, ndv_high,
          nu_low, nu_high,
          eta_low, eta_high
      )
)

p_low, p_high, q_l, fc_low, fc_high = param_to_pq0_PD_jitted(param_star)

wedges = p_high - p_low

print("Low NDVI marginal prices:", p_low)
print("High NDVI marginal prices:", p_high)
print("Price wedges:", wedges)
print("Kinks:", q_l)
print("Fixed payments_low ndvi:", fc_low)
print("Fixed payments_high ndvi:", fc_high)

p_uni, q_uni, fc_uni = param_to_pq0_jitted(param_0)

welfare_uniform, revenue_uniform = get_welfare_and_revenue_jitted(
    context,
    p_uni, q_uni, fc_uni,
    Z_current,
    NDVI,
    nu_array,
    eta_l
)

welfare_low, revenue_low = get_welfare_and_revenue_jitted(
    context_low,
    p_low, q_l, fc_low,
    Z_low, ndv_low,
    nu_low, eta_low
)

welfare_high, revenue_high = get_welfare_and_revenue_jitted(
    context_high,
    p_high, q_l, fc_high,
    Z_high, ndv_high,
    nu_high, eta_high
)

welfare_naive_PD = welfare_low + welfare_high
revenue_naive_PD = revenue_low + revenue_high


print("Uniform welfare:", float(welfare_uniform))
print("Naive PD welfare:", float(welfare_naive_PD))
print("PD welfare:", float(welfare_PD))
print("Welfare gain:", float(welfare_PD - welfare_naive_PD))

print("Uniform revenue:", float(revenue_uniform))
print("Naive PD revenue:", float(revenue_naive_PD))
print("PD revenue:", float(revenue_PD))


#################################
#### Export Results ####
#################################

pd.DataFrame(p_low).to_csv(
    f"{output_dir}/Naive_PD_price_low_revenue_cons.csv",
    index=False
)

pd.DataFrame(p_high).to_csv(
    f"{output_dir}/Naive_PD_price_high_revenue_cons.csv",
    index=False
)

pd.DataFrame(q_l).to_csv(
    f"{output_dir}/Naive_PD_kinks_revenue_cons.csv",
    index=False
)

pd.DataFrame(fc_low).to_csv(
    f"{output_dir}/Naive_PD_fixed_payments_low_revenue_cons.csv",
    index=False
)

pd.DataFrame(fc_high).to_csv(
    f"{output_dir}/Naive_PD_fixed_payments_high_revenue_cons.csv",
    index=False
)

q_low = get_q_sum_hh_jitted(
    context_low.params, context_low.segment,
    p_low, q_l, fc_low,
    Z_low, ndv_low,
    nu_low, eta_low
) / sim

r_low = from_q_to_r_jitted(
    context_low.params,
    context_low.segment,
    q_low,
    p_low, q_l, fc_low
)

cs_low = get_v_out(
    context_low.params,
    context_low.segment,
    q_low,
    p_low, q_l, fc_low,
    Z_low, ndv_low,
    nu_low, eta_low
)

ev_low = get_ev(
    context_low.params,
    context_low.segment,
    q_low,
    p_low, q_l, fc_low,
    Z_low, ndv_low,
    nu_low, eta_low
)


q_high = get_q_sum_hh_jitted(
    context_high.params, context_high.segment,
    p_high, q_l, fc_high,
    Z_high, ndv_high,
    nu_high, eta_high
) / sim

r_high = from_q_to_r_jitted(
    context_high.params,
    context_high.segment,
    q_high,
    p_high, q_l, fc_high
)

cs_high = get_v_out(
    context_high.params,
    context_high.segment,
    q_high,
    p_high, q_l, fc_high,
    Z_high, ndv_high,
    nu_high, eta_high
)

ev_high = get_ev(
    context_high.params,
    context_high.segment,
    q_high,
    p_high, q_l, fc_high,
    Z_high, ndv_high,
    nu_high, eta_high
)


q_full = jnp.zeros_like(segment_current.income)

q_full = q_full.at[low_mask].set(q_low)
q_full = q_full.at[high_mask].set(q_high)

r_full = jnp.zeros_like(segment_current.income)

r_full = r_full.at[low_mask].set(r_low)
r_full = r_full.at[high_mask].set(r_high)

cs_full = jnp.zeros_like(segment_current.income)

cs_full = cs_full.at[low_mask].set(cs_low)
cs_full = cs_full.at[high_mask].set(cs_high)

ev_full = jnp.zeros_like(segment_current.income)

ev_full = ev_full.at[low_mask].set(ev_low)
ev_full = ev_full.at[high_mask].set(ev_high)


hh_df = pd.DataFrame({
    "prem_id": np.array(segment_current.prem_id),
    "segment_id": np.array(segment_current.segment_id),
    "q": np.array(q_full),
    "r": np.array(r_full),
    "cs": np.array(cs_full),
    "ev": np.array(ev_full),
})

hh_df.to_csv(
    f"{output_dir}/Naive_PD_hh_level_results_revenue_cons.csv",
    index=False
)

welfare_total = (
    nansum_ignore_nan_inf_jitted(ev_low * weights_low) +
    nansum_ignore_nan_inf_jitted(ev_high * weights_high)
)


revenue_total = (
    nansum_ignore_nan_inf_jitted(r_low) +
    nansum_ignore_nan_inf_jitted(r_high)
)



summary_df = pd.DataFrame({
    "total_welfare": [float(welfare_total)],
    "total_revenue": [float(revenue_total)],
})

summary_df.to_csv(
    f"{output_dir}/Naive_PD_aggregate_summary_revenue_cons.csv",
    index=False
)
