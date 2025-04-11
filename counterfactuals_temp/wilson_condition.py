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


demand_2018_using_new = pd.read_csv('demand_2018_using_new.csv')

p_l0 = jnp.array([ 3.09,  5.01,  8.54, 12.9 , 14.41])
#[1.92*2, 3.53*4, 4.36*5, 1.51*9]

#[3.84, 21.18, 47.96, 30.2]
#[3.84, 14.12, 21.8, 13.59]
# [2.3, 5.7, 20.5, 0]

q_l0 = jnp.array([ 2,  6, 11, 20])
fc_l0 = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.25+29.75])
#demand_2018_using_new=demand_2018_using_new[demand_2018_using_new['income']<=15000]

demand_2018_using_new_2 = demand_2018_using_new[abs(demand_2018_using_new['quantity'] - 2) <= 0.4]
demand_2018_using_new_6 = demand_2018_using_new[abs(demand_2018_using_new['quantity'] - 6) <= 0.4]
demand_2018_using_new_11 = demand_2018_using_new[abs(demand_2018_using_new['quantity'] - 11) <= 0.4]
demand_2018_using_new_20 = demand_2018_using_new[abs(demand_2018_using_new['quantity'] - 20) <= 0.4]

(fc_l0[1]-fc_l0[0])/(p_l0[1]-p_l0[0])

(fc_l0[2]-fc_l0[1])/(p_l0[2]-p_l0[1])

(fc_l0[3]-fc_l0[2])/(p_l0[3]-p_l0[2])

(fc_l0[4]-fc_l0[3])/(p_l0[4]-p_l0[3])

bins0=25

value = demand_2018_using_new_2['quantity']
bin_size = 0.1
bins = np.arange(value.min(), value.max() + bin_size, bin_size)
value.hist(bins=bins)  # You can adjust the number of bins
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Quantity near 2')
plt.axvline(x=2, color='black', linestyle='--', label='2')
plt.legend()
plt.show()

value = demand_2018_using_new_6['quantity']
bin_size = 0.1
bins = np.arange(value.min(), value.max() + bin_size, bin_size)
value.hist(bins=bins)  # You can adjust the number of bins
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Quantity near 6')
plt.axvline(x=6, color='black', linestyle='--', label='6')
plt.legend()
plt.show()

value = demand_2018_using_new_11['quantity']
bin_size = 0.1
bins = np.arange(value.min(), value.max() + bin_size, bin_size)
value.hist(bins=bins)  # You can adjust the number of bins
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Quantity near 11')
plt.axvline(x=11, color='black', linestyle='--', label='11')
plt.legend()
plt.show()

value = demand_2018_using_new_20['quantity']
bin_size = 0.1
bins = np.arange(value.min(), value.max() + bin_size, bin_size)
value.hist(bins=bins)  # You can adjust the number of bins
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Quantity near 20')
plt.axvline(x=20, color='black', linestyle='--', label='20')
plt.legend()
plt.show()



