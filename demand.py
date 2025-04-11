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


demand_2018 = pd.read_csv('demand_2018.csv')

#demand_2018["essential_usage_mp"] = "Unknown"
#demand_2018["essential_usage_mp"] = demand_2018["essential_usage_mp"].case_when([
 #   (demand_2018.eval("0 <= essential_usage <= 2"), 2.89+0.2), 
  #  (demand_2018.eval("2 < essential_usage <= 6"), 4.81+0.2),  
   # (demand_2018.eval("6 < essential_usage <= 11"), 8.34+0.2),
    #(demand_2018.eval("11 < essential_usage <= 20"), 12.70+0.2),
    #(demand_2018.eval("20 < essential_usage"), 14.21+0.2)
#])

#################
# preperation
#################

fc_l = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.25+29.75])

p_l = jnp.array([2.89+0.2, 4.81+0.2, 8.34+0.2, 12.70+0.2, 14.21+0.2])
p_l_CAP =  jnp.array([2.37+0.05, 4.05+0.05, 6.67+0.05, 11.51+0.05, 14.21+0.05])

q_kink_l = jnp.array([2, 6, 11, 20])

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

demand_2018 = demand_2018.sort_values(by=['prem_id', "bill_ym"])

usage_charge_2018_CAP = pd.read_csv('usage_charge_2018_CAP.csv')

# 215760 of HHs

def get_k(q, q_l):
    conditions_k = [
        (q <q_l[0]),
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
get_k_jitted = jax.jit(get_k)



def expenditure_func(w, q_l, p_l, fc_l):
    bins = jnp.concatenate((jnp.array([0]), q_l, jnp.array([jnp.inf])))
    binned_data = jnp.digitize(w, bins)
    q_plus1_l = jnp.insert(q_l, 0, 0)
    q_diff_l = q_l - q_plus1_l[0:4]
    cumu_sum = jnp.cumsum(p_l[0:4] * q_diff_l)
    result = jnp.where(binned_data==1, fc_l[0] + p_l[0]*w, 
                       fc_l[binned_data-1] + cumu_sum[binned_data-2] + p_l[binned_data-1] * (w - q_l[binned_data-2]))
    return result
expenditure_func_jitted = jax.jit(expenditure_func)

demand_2018 = pd.merge(demand_2018, usage_charge_2018_CAP, on=['prem_id', 'bill_ym'], how = 'left')

demand_2018_CAP = demand_2018[['prem_id', 'CAP_HH', 'CAP']]

demand_2018_CAP = demand_2018_CAP.groupby('prem_id')[['CAP_HH', 'CAP']].max().reset_index()

#CAP_HH
#0.0    126772   0.9956724%
#1.0       551   0.004%

### Overall kept 60% of the HHs, 0.5584397961399138 of the transactions

demand_2018['CAP_HH'] = demand_2018['CAP_HH'].fillna(1)

#demand_2018_CAP['CAP_fare'] = expenditure_func(np.array(demand_2018_CAP['quantity']), q_kink_l, p_l_CAP, fc_l) * np.array(demand_2018_CAP['deflator'])

#demand_2018_CAP_compare = demand_2018_CAP[['prem_id', 'bill_ym', 'charge', 'fare', 'CAP_fare']]

#demand_2018_CAP_compare['CAP_diff'] = abs(demand_2018_CAP_compare['charge'] - demand_2018_CAP_compare['CAP_fare'])

#demand_2018_CAP_compare['mean_CAP_diff'] = demand_2018_CAP_compare.groupby('prem_id')['CAP_diff'].transform('mean')

#demand_2018_CAP_compare =demand_2018_CAP_compare.sort_values(by=['prem_id', "mean_CAP_diff"])

### Only use winter months
#demand_2018_using = demand_2018[ (demand_2018['wastewateravg'] == 0) & (demand_2018['CAP'] == 0)]
demand_2018_using = demand_2018[(demand_2018['CAP'] == 0)]
demand_2018_using = demand_2018

final_ndvi_small = pd.read_csv('prem_key/final_ndvi_small.csv')
final_ndvi_small.rename(columns={'prev_NDVI_final': 'prev_NDVI', 'NDVI_final': 'NDVI'}, inplace=True)

demand_2018_using = pd.merge(demand_2018_using, final_ndvi_small, on=['prem_id', 'bill_ym'], how = 'left')

#2345742 transactions

#demand_2018_winter = demand_2018[ (demand_2018['wastewateravg'] == 1) & (demand_2018['CAP'] == 0)]

### Create a categorical variable to label the size of the lawn:
    ## 0 -5000
    ## 5000 - 6500
    ## 6500 - 8500
    ##  8500 - 10000
    ## > 10000

lawn_bins = [0, 5000, 6500, 8500, 10000, np.inf]
lawn_cat = np.digitize(demand_2018_using['lawn_area'], lawn_bins)
demand_2018_using['lawn_cat'] = lawn_cat
demand_2018_using['total_area'] = demand_2018_using['house_area'] + demand_2018_using['lawn_area']
demand_2018_using['lawn_percentage'] = demand_2018_using['lawn_area'] / demand_2018_using['total_area']

### Create a categorical variable to label the proportion of the lawn:
    ## 0 -0.75
    ## 0.75 - 0.8
    ## 0.8 - 0.9
    ##  0.9 - 1

lawn_p_bins = [0, 0.75, 0.8, 0.9, 1]
demand_2018_using['lawn_prop_cat']= np.digitize(demand_2018_using['lawn_percentage'], lawn_p_bins)

#lawn_areaxNDVI = pd.DataFrame({'numerical':  demand_2018_using['previous_NDVImyd_diff'],
 #       'categorical': demand_2018_using['lawn_cat']})

#lawn_areaxNDVI = pd.get_dummies(lawn_areaxNDVI, columns=['categorical'], drop_first=True)

# Create interaction terms
#interaction = PolynomialFeatures(degree=1, interaction_only=True, include_bias=False)
#interaction_terms = interaction.fit_transform(df[['numerical', 'categorical_B']])

# Add interaction terms to the DataFrame
#interaction_df = pd.DataFrame(interaction_terms, columns=['numerical', 'categorical_B', 'interaction_term'])\
    

#demand_2018_using['lawn_areaxNDVI'] = jnp.multiply(jnp.array(demand_2018_using['lawn_area'])/43560, jnp.array( demand_2018_using['previous_NDVImyd']))
#demand_2018_using['lawn_areaxNDVI'] = demand_2018_using['lawn_areaxNDVI']/43560 ## change the unit to acerage

#demand_2018_using['lawn_areaxTmax']= jnp.multiply(jnp.array(demand_2018_using['lawn_area']), jnp.array( demand_2018_using['mean_TMAX_1']))
#demand_2018_using['lawn_areaxTmax'] = demand_2018_using['lawn_areaxTmax']/43560

#demand_2018_using['lawn_areaxPRCP']= jnp.multiply(jnp.array(demand_2018_using['lawn_area']), jnp.array( demand_2018_using['total_PRCP']))
#demand_2018_using['lawn_areaxPRCP'] = demand_2018_using['lawn_areaxPRCP']/43560

def categorize_total_area(value):
    if value < 43560: ## 1 acerage
        return 0
    else:
        return 1

demand_2018_using['above_one_acre'] = demand_2018_using['total_area'].apply(categorize_total_area)

conditions = [
    (demand_2018_using['bill_ym'] <= 201806),
    (demand_2018_using['bill_ym'] > 201806) & (demand_2018_using['bill_ym'] <=201807),
    (demand_2018_using['bill_ym'] > 201807) & (demand_2018_using['bill_ym'] <=201808),
    (demand_2018_using['bill_ym'] > 201808) & (demand_2018_using['bill_ym'] <=201809),
    (demand_2018_using['bill_ym'] > 201809) & (demand_2018_using['bill_ym'] <=201903),
    (demand_2018_using['bill_ym'] > 201903) & (demand_2018_using['bill_ym'] <=201904),
    (demand_2018_using['bill_ym'] > 201904) & (demand_2018_using['bill_ym'] <=201907),
    (demand_2018_using['bill_ym'] > 201907) & (demand_2018_using['bill_ym'] <=201908),
    (demand_2018_using['bill_ym'] > 201908) & (demand_2018_using['bill_ym'] <=201909),
    (demand_2018_using['bill_ym'] > 201909) & (demand_2018_using['bill_ym'] <=201910),
    (demand_2018_using['bill_ym'] > 201910) 
]

# Define corresponding values
#choices = [0, 1, 2, 1, 0, 1, 0, 1, 3, 4, 2]
choices = [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
# Create new column based on conditions
demand_2018_using.loc[:, 'drought_stage'] = np.select(conditions, choices, default=0)


formula = 'np.log(quantity) ~ hvac_residential + prev_NDVI + spa + heavy_water_app+ mean_TMAX_1 + IQR_TMAX_1 + total_PRCP + IQR_PRCP+drought_stage'
#formula = 'np.log(quantity) ~ mean_TMAX_1 + IQR_TMAX_1 + total_PRCP + IQR_PRCP +lawn_areaxNDVI +prv_year_avg_essential_usage + heavy_water_spa + np.log(previous_essential_usage_mp) + np.log(income)'
#X = sm.add_constant(X)
#y = jnp.log(demand_2018_using['quantity'])
#l_model = sm.OLS(y, X)
l_model = sm.OLS.from_formula(formula, data = demand_2018_using)
results = l_model.fit()
print(results.summary())

#X = np.column_stack( (
    #np.log(demand_2018_using['hvac_residential']), 
                      #demand_2018_using['heavy_water_app'], 
 #                     demand_2018_using['bedroom'], 
                      #demand_2018_using['bathroom'],
                      #demand_2018_using['lawn_area'], demand_2018_using['previous_NDVImyd_diff'],  
                      #demand_2018_using['lawn_area']* demand_2018_using['previous_NDVImyd_diff'],
  #                    demand_2018_using['mean_TMAX_1'],
                      #demand_2018_using['IQR_TMAX_1'], demand_2018_using['lake_level'],
                      #demand_2018_using['total_PRCP'],
                      #demand_2018_using['IQR_PRCP'],
                    #np.log(demand_2018_using['previous_avg_p_water']
   #                 demand_2018_using['income'],
    #                       demand_2018_using['prv_year_avg_essential_usage'], demand_2018_using['essential_usage_mp'], demand_2018_using['essential_usage_ap'],
     #                     ) )

#X = pd.DataFrame(X, 
 #  columns=[
       #'hvac_residential', 
  #          "heavy_water_app","bedroom", "bathroom", 
   #         "lawn_area", "previous_NDVImyd_diff",
            #"lawn_areaxNDVI", 
    #        "mean_TMAX_1",
            #"IQR_TMAX_1", "lake_level",
     #       "totalPRCP",
            #"IQR_PRCP", 
            #"previous_avg_p_water"
      #      "income",
       #     "prv_year_avg_essential_usage", "essential_usage_mp", "essential_usage_ap"
        #    ])
#vif_data = pd.DataFrame()
#vif_data["Variable"] = X.columns
#vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
#print(vif_data)

#corr_matrix = X.corr()

# Plot the correlation matrix as a heatmap
#print("Correlation Matrix:")
#print(corr_matrix)

#demand_2018_temp = pd.DataFrame(jnp.column_stack((results.fittedvalues, results.resid, demand_2018_winter)))
#demand_2018_temp = demand_2018_temp[ (demand_2018_temp.iloc[:, 1]>=1.5 )]

plt.scatter(demand_2018_using['total_PRCP'], np.log(demand_2018_using['quantity']))
#plt.xlabel('Fitted Values')
#plt.ylabel('Residuals')
#plt.title('Fitted Values vs Residuals Plot')
#plt.grid(True)
#plt.axhline(y=0, color='k', linestyle='--')  # Add horizontal line at y=0
plt.show()

#beta_8 = results.params[1]
#alpha_0 = results.params[4]

#X_using = jnp.column_stack( (jnp.log(demand_2018_using['hvac_residential']), demand_2018_using['mean_TMAX_1'],
 #                     demand_2018_using['IQR_TMAX_1'],
  #                  jnp.log(demand_2018_using['essential_usage_ap'])) )
#X_using = sm.add_constant(X_using)
#w_0_current =  jnp.exp(results.predict(X_using))

demand_2018_using = demand_2018_using[(demand_2018_using['bedroom'] <= 200)]
demand_2018_using = demand_2018_using[(demand_2018_using['bathroom'] <= 200)]

demand_2018_using['bedroombathroom'] = demand_2018_using['bedroom'] + demand_2018_using['bathroom']
demand_2018_using['hvac_residential'] = demand_2018_using['hvac_residential']/43560
demand_2018_using['lawn_area'] = demand_2018_using['lawn_area']/43560

demand_2018_using = demand_2018_using.dropna()

### 2351626 transaction   ## 127320 HHs

demand_2018_using.to_csv('demand_2018_using.csv', index=False)

demand_2018_using = pd.read_csv('demand_2018_using.csv')

#### Start Demand Estimation

def scale_array (arr):
    arr_after = jnp.where(arr > 1, jnp.log(arr) + 1, arr)  
    return arr_after

def mode(array):
    unique_values, counts = jnp.unique(array, return_counts=True)
    max_count_index = jnp.argmax(counts)
    return unique_values[max_count_index]

scale_array_jitted = jax.jit(scale_array)

A_using_df = demand_2018_using[['heavy_water_app','spa', 'bathroom','bedroom', 'prev_NDVI', 'drought_stage','above_one_acre', 'income']]

from statsmodels.stats.outliers_influence import variance_inflation_factor

import numpy as np
import matplotlib.pyplot as plt

# Plot histogram
plt.hist(np.square(np.log(np.array(demand_2018_using['income']))), bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Data')
plt.show()


# Compute VIF for each column
X = A_using_df.copy()  # Feature matrix
X['Intercept'] = 1  # Add intercept for VIF computation
vif_data = pd.DataFrame({
    'Feature': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})

# Drop Intercept row
vif_data = vif_data[vif_data['Feature'] != 'Intercept']

print(vif_data)

demand_2018_using['lawn_areaxNDVI'] = jnp.multiply(jnp.array(demand_2018_using['lawn_area'])/43560, jnp.array( demand_2018_using['prev_NDVI']))


A_current_outdoor = jnp.column_stack(( 
    #jnp.array(demand_2018_using['heavy_water_spa']), 
    #jnp.array(demand_2018_using['spa']), 
    #jnp.array(demand_2018_using['heavy_water_area']), 
    #jnp.array(demand_2018_using['spa_area']), 
    jnp.array(demand_2018_using['bathroom']), 
                                      #jnp.array(demand_2018_using['lawn_areaxNDVI']), 
                                      jnp.array( demand_2018_using['prev_NDVI']),
                                      #jnp.log(jnp.array(demand_2018_using['lawn_area'])),
                                      #jnp.array(demand_2018_using['above_one_acre']),
                                      #jnp.array(demand_2018_using['house_area'])
                                      # jnp.array(demand_2018_using['drought_stage']),
                                      ))
#A_current_outdoor = jnp.log(jnp.array(demand_2018_using['bedroom']))
#A_current_outdoor = A_current_outdoor[:, jnp.newaxis]
A_current_indoor = jnp.column_stack((jnp.array(demand_2018_using['bathroom']),
                                     #jnp.array(demand_2018_using['bedroom'])
                                     #jnp.array(demand_2018_using['prv_year_avg_essential_usage'])
                                     jnp.array(demand_2018_using['above_one_acre'])
                                       ))
#A_current_indoor = jnp.array(demand_2018_using['bathroom'])
#A_current_indoor = A_current_indoor[:, jnp.newaxis]
A_current_price = jnp.column_stack((
    #scale_array_jitted(jnp.array(demand_2018_using['heavy_water_spa_area'])), 
    #jnp.array(demand_2018_using['heavy_water_app']), 
    #jnp.array( demand_2018_using['previous_NDVImyd']),
    jnp.array(demand_2018_using['bedroom']), 
    jnp.array(demand_2018_using['prev_NDVI']), 
    jnp.array(demand_2018_using['mean_TMAX_1']),
    jnp.array(demand_2018_using['total_PRCP'])
    ))

A_current_income = jnp.column_stack((
    #scale_array_jitted(jnp.array(demand_2018_using['heavy_water_spa_area'])), 
    jnp.array(demand_2018_using['heavy_water_app']), 
    jnp.array(demand_2018_using['bedroom']), 
    #jnp.array( demand_2018_using['previous_NDVImyd']),
    jnp.array(demand_2018_using['prev_NDVI']), 
    #jnp.array(demand_2018_using['mean_TMAX_1']),
    #jnp.array(demand_2018_using['total_PRCP'])
    ))

Z_current_outdoor = jnp.column_stack((jnp.array(demand_2018_using['mean_TMAX_1']),
                                      #jnp.array(demand_2018_using['lawn_areaxTmax']),
                                      jnp.array(demand_2018_using['IQR_TMAX_1']),
                                      jnp.array(demand_2018_using['total_PRCP']) 
                                      ,jnp.array(demand_2018_using['IQR_PRCP'])
                                      #,jnp.array(demand_2018_using['lawn_areaxPRCP'])
                                      ))
#Z_current_outdoor = jnp.array(demand_2018_using['mean_TMAX_1'])
#Z_current_outdoor = Z_current_outdoor[:, jnp.newaxis]
Z_current_indoor = jnp.array(demand_2018_using['mean_TMAX_1'])
Z_current_indoor = Z_current_indoor[:, jnp.newaxis]
#Z_current_indoor = jnp.column_stack((jnp.array(demand_2018_using['mean_TMAX_1'])
 #                                    ,jnp.array(demand_2018_using['total_PRCP'])
  #                                   ))
G = jnp.array(demand_2018_using['prev_NDVI'])
I = jnp.array(demand_2018_using['income'])
p0 = jnp.array(demand_2018_using['previous_essential_usage_mp'])
w_i = jnp.array(demand_2018_using['quantity'])
#w_0_current = jnp.exp(results.fittedvalues)

def get_total_wk (beta_1, beta_2,
                  #beta_3, 
                  c_wo,
                   beta_4, 
                  #beta_5,
                  c_a,
                  #alpha, 
                  beta_6, 
                  #beta_7,
                  c_r,
                  #rho, 
                  k, 
                  #beta_8, 
                  #beta_9, 
                  #alpha_0,
                  #c_wi,
                  A_o = A_current_outdoor, Z_i = Z_current_indoor, Z_o = Z_current_outdoor, 
                  A_p = A_current_price, 
                  A_i = A_current_income,
                  CAP = jnp.array(demand_2018_using['CAP_HH']),
                  G = jnp.array(demand_2018_using['previous_NDVImyd_diff']),
                  p = p_l, I = jnp.array(demand_2018_using['income']),
                  p0 = jnp.array(demand_2018_using['previous_essential_usage_mp']), 
                  de = jnp.array(demand_2018_using['deflator']),
                  #w_0 = w_0_current
                  ):
    p_k = jnp.where(CAP == 1, p_l_CAP[k], p_l[k])
    #p_k = p[k]
    d_k = jnp.where(CAP == 1, calculate_dk_CAP_jitted(k), calculate_dk_jitted(k))
    alpha = abs(
     jnp.exp(
    jnp.dot(A_p, beta_4) +
    # jnp.array(beta_5 * G) +
    c_a
     )
    )
    #alpha = jnp.exp(jnp.dot(A, beta_4) )
    #rho = jnp.exp(jnp.dot(A, beta_6) )
    rho =abs(
        # jnp.exp(
        jnp.dot(A_i, beta_6) +
                 # + jnp.array(beta_7*G)
                  c_r
                 # )
                 )
    #alpha = jnp.exp(alpha)
    #rho = jnp.exp(rho)
    #alpha_0 = abs(alpha_0)
    w_outdoor = jnp.exp(jnp.dot(A_o, beta_1) + jnp.dot(Z_o, beta_2)
                       #+ jnp.array(beta_3*G)
                       - jnp.multiply( jnp.multiply(alpha, jnp.log(p_k)), de) + 
                       jnp.multiply(rho, jnp.log(jnp.maximum(I+ jnp.multiply(d_k, de), 1e-16))) + c_wo)
    #w_indoor = w_0_current
   # w_indoor = jnp.exp(jnp.dot(A_i, beta_8) 
    #                   + jnp.dot(Z_i, beta_9)
                       #-alpha_0*jnp.log(p0)
     #                  + c_wi
      #                 )
    #result = jnp.log(w_outdoor + w_indoor)
    result = jnp.log(w_outdoor)
    return result

get_total_wk_jitted = jax.jit(get_total_wk)

def norm_pdf(x, mean=0.0, stddev=1.0):
    return (1.0 / jnp.sqrt(2 * jnp.pi * stddev**2)) * jnp.exp(-0.5 * ((x - mean) / stddev)**2)
norm_pdf_jitted = jax.jit(norm_pdf)

def norm_cdf(x, mu=0.0, sigma=1.0):
    return 0.5 * (1 + erf((x - mu) / (sigma * jnp.sqrt(2))))
norm_cdf_jitted = jax.jit(norm_cdf)

def likelihood_f_base (beta, A_i = A_current_indoor, A_o = A_current_outdoor, Z_i=Z_current_indoor, Z_o=Z_current_outdoor, 
                  #A = A_current,
                  G = jnp.array(demand_2018_using['prev_NDVI']),
                  p = p_l, I = jnp.array(demand_2018_using['income']),
                  p0 = jnp.array(demand_2018_using['previous_essential_usage_mp']), w_i = jnp.array(demand_2018_using['quantity']),
                  #w_0 = w_0_current
                  ):
    b1_1 = jnp.exp(beta[0])
    #b1_1 = 3.4906594e-03
    b1_2 = jnp.exp(beta[1])
    #b1_3 = -1*jnp.exp(beta[2])
    #b1_4 = -1*jnp.exp(beta[3])
    #b1_5 = -1*jnp.exp(beta[4])
    #b1_6 = -1*jnp.exp(beta[5])
    #b1_7 = beta[6]
    b2_1 = beta[2]
    b2_2 = beta[3]
    b2_3 = -1*jnp.exp(beta[4])
    b2_4 = beta[5]
    #b3 = beta[2]
    c_o = beta[6]
    b4_1 = beta[7]
    b4_2 = beta[8]
    b4_3 = beta[9]
    b4_4 = beta[10]
    #b4_5 = beta[11]
    #b5 = beta[7],  1.35596190e-03, -3.14184487e-01, -3.89749146e+00,
    c_alpha = beta[11]
    #c_alpha = jnp.minimum((beta[11]), 1)
    b6_1 = beta[12]
    b6_2 = beta[13]
    b6_3 = beta[14]
    #b6_4 = beta[19]
    #b6_5 = beta[20]
    #b7 = beta[10]
    c_rho = beta[15]
    #a = beta[11]
    #r = beta[15]
    sigma_eta = jnp.sqrt(jnp.square(beta[16]))
    sigma_nu = jnp.sqrt(jnp.square(beta[17]))
    #b8_1 = 2.21E-04
    #b8_2= -4.18E-04
    #b8_3 = beta[17]
    #b9_1 = 2.98E-03
    #b9_2 = beta[18]
    #a0 = beta[18]
    #c_i = 3.85E-01
    #k = jnp.array([0, 1, 2, 3, 4])
    q_k = jnp.array([2, 6, 11, 20, 1e+16])
    sigma = jnp.square(sigma_eta)/ (jnp.sqrt(jnp.square(sigma_eta) * (jnp.square(sigma_eta) + jnp.square(sigma_nu)))) 
    def get_total_wk_k (k):
        result = get_total_wk_jitted(beta_1 = jnp.array([b1_1, b1_2]), 
                                     beta_2 = jnp.array([b2_1, b2_2, b2_3, b2_4]), 
                                     #beta_3 = b3,
                                     c_wo = c_o,
                       beta_4 = jnp.array([b4_1, b4_2, b4_3, b4_4]), 
                       #beta_5 = b5
                       c_a = c_alpha,
                       beta_6 = jnp.array([b6_1, b6_2, b6_3]),
                       c_r = c_rho,
                       #alpha = a, 
                       #rho = r,
                       #beta_8 = jnp.array([b8_1, b8_2]), 
                       #beta_9 = jnp.array([b9_1]),
                       #alpha_0=a0,
                       #c_wi = c_i,
                       k=k)
        return result
    get_total_wk_k_jitted = jax.jit(get_total_wk_k)

    log_w = jnp.column_stack((get_total_wk_k_jitted(0), get_total_wk_k_jitted(1), get_total_wk_k_jitted(2),
                    get_total_wk_k_jitted(3), get_total_wk_k_jitted(4)))
    #log_w_first = log_w[:, :4]
    t_k = (-1*log_w + jnp.array(jnp.log(q_k))[jnp.newaxis, :])/sigma_eta
    s_k = (-1*log_w + jnp.array(jnp.log(w_i))[:, jnp.newaxis]) / (jnp.sqrt(jnp.square(sigma_eta)+ jnp.square(sigma_nu) )) 
    m_k0 = (-1 * log_w +  jnp.array(jnp.log(jnp.insert(q_k, 0, 1e-16)[:-1]))[jnp.newaxis, :])/sigma_eta
    m_k = m_k0[:, 1:5]
    u_k = (jnp.tile(jnp.log(w_i), (4, 1)).T - jnp.array(jnp.log(q_k[:-1]))[jnp.newaxis, :])/sigma_nu
    n_k = (m_k0 - sigma*s_k)/ jnp.sqrt(1-jnp.square(sigma))
    r_k = (t_k - sigma*s_k) / jnp.sqrt(1-jnp.square(sigma))
    phi_sk = norm_pdf_jitted(s_k)
    Phi_rk = norm_cdf_jitted(r_k)
    Phi_nk = norm_cdf_jitted(n_k)
    phi_uk = norm_pdf_jitted(u_k)
    Phi_mk = norm_cdf_jitted(m_k)
    Phi_tk = norm_cdf_jitted(t_k[:, 1:6])
    likelihood =jnp.maximum( jnp.sum(jnp.nan_to_num(jnp.multiply(phi_sk / jnp.sqrt(jnp.square(sigma_eta) + jnp.square(sigma_nu)), 
                           (Phi_rk - Phi_nk)), nan = 0) , axis=1) + 
    jnp.sum(jnp.nan_to_num(jnp.multiply( phi_uk / sigma_nu, 
                           (Phi_mk -Phi_tk)), nan = 0) , axis=1), 1e-16)
    fit = jnp.log(likelihood)
    fit_value = jax.device_get(fit)
    return fit_value

def ll_to_sumll (func):
    def wrapper(*args, **kwargs):
        log_ll = func(*args, **kwargs)
        fit = -1*jnp.sum(log_ll)
        fit_value = jax.device_get(fit)
        #print("Current Negative Likelihood:")
        jax.debug.print("Current Negative Likelihood {x}", x= fit_value)
        return fit_value
    return wrapper

likelihood_f_base_jitted = jax.jit(likelihood_f_base)

@ll_to_sumll
def likelihood_f (beta):
    return likelihood_f_base_jitted(beta)

likelihood_f_jitted = jax.jit(likelihood_f)

#### Estimation
shape = (19, )

rng_key = random.PRNGKey(100)  # You can change the seed (second argument) for different random numbers
#starting = jax.random.uniform(rng_key, shape)
#starting = jax.random.normal(rng_key, shape)
#starting = jnp.repeat(0.01, 16)
starting = jnp.array([0, 0,
                      0, 0, 0, 0,
                      1, 
                      0, 0, 0, 0,0, 
                      1, 
                      0,0, 0,
                      1, 
                      1, 0.01
                      ])

# Define the parameters for the sequence
num_matrices = 100
matrix_shape = (19, )  # Example shape, you can adjust as needed

# Create a sequence of random matrices
matrix_sequence = [starting + np.random.normal(*matrix_shape)*0.0001 for _ in range(num_matrices)]
starting_grid = jnp.vstack(matrix_sequence)

starting_result = np.apply_along_axis(likelihood_f_jitted, axis=1, arr=starting_grid)
starting = starting_grid[np.argmin(starting_result)]


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

'''

"""""""
### 16: Ao = heavywater+log(lawn), Ai = bedroom, Zo - meanT+totalprcp, Zi =- meanT
#result16_1 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead')
### 17: Ao = spa + log(lawn), fix p0 = essential_water_ap
#result17_1 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 5000})
### 18: Ao = spa + lawn, fix p0 = essential_water_mp
#result18_1 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 5000})
### 19: Ao = spa + lawn, fix p0 = essential_water_ap
#result19_1 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 5000})
### 20: Ao = spa/heavywater + lawn, Ai = bathroom, fix p0 = previous_essential_water_ap
#result20_1 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 5000})
### 20-1, spa/heavy water + log(lawn)
#result20_1_1 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 5000})

### 20-2, spa/heavy water + log(lawn), Ai = bedroom, p0 = essential_water_ap
#result20_1_2 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 5000})
#result20_1_2_2 = minimize(likelihood_f_jitted, result20_1_2.x, method = 'Nelder-Mead', options={'maxfev': 5000})
######### result 20*(on excel) = 20_1_2_2 is the best one so far

### 21: add alpha equation and beta equation on top of 20-2 (alpha and beta equation does not have G), use a different A vector
### A = lawn_areaxNDVI + bathroom, Ai = bedroom, p0 = essential_water_ap
#result21_1 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 5000})

### 22: Ao = heavy_water + log(lawn_area), Zo = mean Tmax + IQR_Tmax + total PRCP + IQR_PRCP
### A = lawn_areaxNDVI + bathroom, Ai = bedroom, Zi = mean Tmax + IQR_Tmax, p0 = essential_water_ap
#result22_1 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 5000})
#result22_2 = minimize(likelihood_f_jitted, result22_1.x, method = 'Nelder-Mead', options={'maxfev': 5000})
## se for heavy_water huge, need to add spa

### 23: Ao = heavy_water_spa + log(lawn_area), Zo = mean Tmax + IQR_Tmax + total PRCP + IQR_PRCP
### A = lawn_areaxNDVI + bathroom, Ai = bedroom, Zi = mean Tmax + IQR TMax, p0 = essential_water_ap
#result23_1 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 5000})

### 24: Ao = heavy_water_spa + log(lawn_area), Zo = mean Tmax + IQR_Tmax + total PRCP + IQR_PRCP
### A = lawn_areaxNDVI + bathroom, Ai = bedroom, Zi = mean Tmax + IQR TMax, p0 = previous_essential_water_mp
#result24_1 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})

### 25: Ao = heavy_water + log(lawn_area), Zo = mean Tmax + IQR_Tmax + total PRCP + IQR_PRCP
### A = lawn_areaxNDVI + bathroom, Ai = bedroom, Zi = mean Tmax + IQR TMax, p0 = previous_essential_water_mp
#result25_1 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})

### 25: Ao = heavy_water_spa + log(lawn_area), Zo = mean Tmax + IQR_Tmax + total PRCP + IQR_PRCP
#result25_2 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})

### 25: Ao = spa + log(lawn_area), Zo = mean Tmax + IQR_Tmax + total PRCP + IQR_PRCP
#result25_3 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})

### 25: Ao = heavy_water + spa + log(lawn_area), Zo = mean Tmax + IQR_Tmax + total PRCP + IQR_PRCP
#result25_4 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})

### add constant delete NDVI, estimate alpha and rho seperately, 
##random uniform
#result25_5 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_5_2 = minimize(likelihood_f_jitted, result25_5.x, method = 'Nelder-Mead', options={'maxfev': 10000})
##random normal
#result25_5_1 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_5_1_2 = minimize(likelihood_f_jitted, result25_5_1.x, method = 'Nelder-Mead', options={'maxfev': 10000})
##all 0.01
#result25_5_2 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
###edited 0.01, this has the lowest value, 
#result25_5_3 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_5_3_2 = minimize(likelihood_f_jitted, result25_5_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### add bothspa + heavy water + sweep on starting value 
#result25_6 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_6_2 = minimize(likelihood_f_jitted, result25_6.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_6_3 = minimize(likelihood_f_jitted, result25_6_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### spa + sweep on starting value 
#result25_7 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_7_2 = minimize(likelihood_f_jitted, result25_7.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_7_3 = minimize(likelihood_f_jitted, result25_7_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
### alpha is tiny, rho is also not big, indoor temp is negative, indoor price effect is basically zero

### spa, no p0
#result25_8 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_8_2 = minimize(likelihood_f_jitted, result25_8.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_8_3 = minimize(likelihood_f_jitted, result25_8_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### heavy water/spa,  has IQR for outdoor, has p0, no IQR for indoor
#result25_9 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_9_2 = minimize(likelihood_f_jitted, result25_9.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_9_3 = minimize(likelihood_f_jitted, result25_9_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### heavy_water_spa area + log(lawn area),  has IQRs for outdoor, has p0, no IQR for indoor
#result25_10 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_10_2 = minimize(likelihood_f_jitted, result25_10.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_10_3 = minimize(likelihood_f_jitted, result25_10_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_10_4 = minimize(likelihood_f_jitted, result25_10_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### heavy_water_spa area + lawn area category,  has IQRs for outdoor, has p0, no IQR for indoor
#result25_11 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_11_2 = minimize(likelihood_f_jitted, result25_11.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_11_3 = minimize(likelihood_f_jitted, result25_11_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_11_4 = minimize(likelihood_f_jitted, result25_11_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### heavy_water_spa area + lawn area category, no IQR for outdoor has p0, IQR for indoor, A_i : bathroom
#result25_12 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_12_2 = minimize(likelihood_f_jitted, result25_12.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_12_3 = minimize(likelihood_f_jitted, result25_12_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result25_12_4 = minimize(likelihood_f_jitted, result25_12_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### no IQR, heavy_water_spa area + lawn area category,A_i bathroom, add formula for alpha and rho, use bedroom in A
#result_26_1 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_1_2 = minimize(likelihood_f_jitted, result_26_1.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_1_3 = minimize(likelihood_f_jitted, result_26_1_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_1_4 = minimize(likelihood_f_jitted, result_26_1_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### A_outdoor: heavy_water_spa + log(lawn area) + house > 1 acre, A_indoor: bathroom
#### add formula for alpha, but not rho, A:  heavy_water + lawn_area + constant, no p0
### Z outdoor, T + IQR + PRCP, Z infoor, T
#result_26_2 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_2_2 = minimize(likelihood_f_jitted, result_26_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_2_3 = minimize(likelihood_f_jitted, result_26_2_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_2_4 = minimize(likelihood_f_jitted, result_26_2_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### A_outdoor: heavy_water_spa + lawn_areaxNDVI + house > 1 acre, A_indoor: bathroom + house > 1 acre
#### add formula for alpha, but not rho, A:  heavy_water + lawn_areaxNDVI + constant, no p0
### Z outdoor, T + IQR + PRCP, Z infoor, T
#result_26_3 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_3_2 = minimize(likelihood_f_jitted, result_26_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_3_3 = minimize(likelihood_f_jitted, result_26_3_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_3_4 = minimize(likelihood_f_jitted, result_26_3_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### A_outdoor: scaled_heavy_water_spa_area + lawn_areaxNDVI + house > 1 acre, A_indoor: bathroom + house > 1 acre
#### add formula for alpha, but not rho, A:  heavy_water + lawn_areaxNDVI + constant, no p0
### Z outdoor, T + IQR + PRCP, Z infoor, T
#result_26_4 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_4_2 = minimize(likelihood_f_jitted, result_26_4.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_4_3 = minimize(likelihood_f_jitted, result_26_4_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_4_4 = minimize(likelihood_f_jitted, result_26_4_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### A_outdoor: heavy_water_spa_area + lawn_areaxNDVI + house > 1 acre, A_indoor: hvac residential
#### add formula for alpha, but not rho, A:  heavy_water spa area + lawn_areaxNDVI + constant, no p0
### Z outdoor, T + IQR + PRCP, Z infoor, T + IQR
#result_26_5 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_5_2 = minimize(likelihood_f_jitted, result_26_5.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_5_3 = minimize(likelihood_f_jitted, result_26_5_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_5_4 = minimize(likelihood_f_jitted, result_26_5_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### A_outdoor: heavy_water_spa + lawn_areaxNDVI + house > 1 acre, A_indoor: bathroom
#### add formula for alpha, but not rho, A:  heavy_water spa + lawn_areaxNDVI + constant, no p0
### Z outdoor, T + IQR + PRCP + IQR, Z infoor, T
#result_26_6 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_6_2 = minimize(likelihood_f_jitted, result_26_6.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_6_3 = minimize(likelihood_f_jitted, result_26_6_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_6_4 = minimize(likelihood_f_jitted, result_26_6_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### A_outdoor: heavy_water_spa + lawn_areaxNDVI + house > 1 acre, A_indoor: bathroom + house > 1  acre
#### add formula for alpha, but not rho, A:  heavy_water spa + lawn_areaxNDVI + constant, no p0
### Z outdoor, T + IQR + PRCP + IQR, Z infoor, T + IQR
#result_26_7 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_7_2 = minimize(likelihood_f_jitted, result_26_7.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_7_3 = minimize(likelihood_f_jitted, result_26_7_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_7_4 = minimize(likelihood_f_jitted, result_26_7_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### A_outdoor: heavy_water_spa + lawn_areaxNDVI + house > 1 acre, A_indoor: bathroom + house > 1  acre
#### add formula for alpha, but not rho, A:  heavy_water_spa_area + lawn_areaxNDVI + constant, no p0
### Z outdoor, T + IQR + PRCP + IQR, Z infoor, T + IQR
#result_26_8 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_8_2 = minimize(likelihood_f_jitted, result_26_8.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_8_3 = minimize(likelihood_f_jitted, result_26_8_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_8_4 = minimize(likelihood_f_jitted, result_26_8_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### A_outdoor: scaled_heavy_water_spa_area + lawn_areaxNDVI, A_indoor: bathroom + house>1acerage + p0
#### add formula for alpha, but not rho, A:  heavy_water_spa + lawn_areaxNDVI + housse>1acerage + constant
### Z outdoor, T + IQR + PRCP+IQR, Z indoor, T + IQR
### lawn_areaxNDVI = -1~1 previous NDVI_diff x lawn_area in acerage
#result_26_9 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_9_2 = minimize(likelihood_f_jitted, result_26_9.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_9_3 = minimize(likelihood_f_jitted, result_26_9_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_26_9_4 = minimize(likelihood_f_jitted, result_26_9_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### A_outdoor: heavy_water_app_area + lawn_areaxNDVI + house>1acerage, A_indoor: bathroom + house>1acerage 
#### add formula for alpha, not rho, A:  heavy_water_app+lawn_areaxNDVI +bathroom + constant
### Z outdoor, T + IQR+ PRCP+IQR, Z indoor = T
### lawn_areaxNDVI = -1~1 previous NDVI_diff x lawn_area in sqft
### Use all demand_2018, including winter
#result_27_1 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_27_1_2 = minimize(likelihood_f_jitted, result_27_1.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_27_1_3 = minimize(likelihood_f_jitted, result_27_1_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_27_1_4 = minimize(likelihood_f_jitted, result_27_1_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### A_outdoor: heavy_water_app_area + lawn_areaxNDVI + house>1acerage 
#### add formula for alpha, not rho, A:  heavy_water_app+lawn_areaxNDVI +bathroom + constant
### Z outdoor, T + IQR+ PRCP+IQR, sign bound the Temperature
### lawn_areaxNDVI = -1~1 previous NDVI_diff x lawn_area in sqft
### Use all demand_2018, including winter
### Use result 27_1 for indoor water
#result_28_1 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_28_1_2 = minimize(likelihood_f_jitted, result_28_1.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_28_1_3 = minimize(likelihood_f_jitted, result_28_1_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_28_1_4 = minimize(likelihood_f_jitted, result_28_1_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

## add bound for c_alpha to be <=1
#result_28_2 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_28_2_2 = minimize(likelihood_f_jitted, result_28_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_28_2_3 = minimize(likelihood_f_jitted, result_28_2_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_28_2_4 = minimize(likelihood_f_jitted, result_28_2_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### Z outdoor, T + lawnxTmax + PRCP + lawnxPRCP
#result_28_3 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_28_3_2 = minimize(likelihood_f_jitted, result_28_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_28_3_3 = minimize(likelihood_f_jitted, result_28_3_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_28_3_4 = minimize(likelihood_f_jitted, result_28_3_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### all lanwx use acreage as unit, 
### A_outdoor: heavy_water_app_area + lawn_areaxNDVI + spa_area
#result_28_4 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_28_4_2 = minimize(likelihood_f_jitted, result_28_4.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_28_4_3 = minimize(likelihood_f_jitted, result_28_4_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_28_4_4 = minimize(likelihood_f_jitted, result_28_4_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### all lanwx use sqft as unit, 
### A_outdoor: heavy_water_spa_area + lawn_areaxNDVI + above_oe_acerage, A:heavy_water_spa+lawnxNDV+bathroom + above_one_acerage
#result_28_5 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_28_5_2 = minimize(likelihood_f_jitted, result_28_5.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_28_5_3 = minimize(likelihood_f_jitted, result_28_5_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
#result_28_5_4 = minimize(likelihood_f_jitted, result_28_5_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})



### Correct formula, no indoor vs outdoor, add income formula
### A:heavy_water_spa_area+lawnxNDVI+bathroom + above_one_acerage + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI+bathroom + above_one_acerage
result_29 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
result_29_2 = minimize(likelihood_f_jitted, result_29.x, method = 'Nelder-Mead', options={'maxfev': 10000})
result_29_3 = minimize(likelihood_f_jitted, result_29_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
result_29_4 = minimize(likelihood_f_jitted, result_29_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### A:heavy_water_spa+lawnxNDVI+bathroom + above_one_acerage + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI+bathroom + above_one_acerage + drought_stage
result_30 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 10000})
result_30_2 = minimize(likelihood_f_jitted, result_30.x, method = 'Nelder-Mead', options={'maxfev': 10000})
result_30_3 = minimize(likelihood_f_jitted, result_30_2.x, method = 'Nelder-Mead', options={'maxfev': 10000})
result_30_4 = minimize(likelihood_f_jitted, result_30_3.x, method = 'Nelder-Mead', options={'maxfev': 10000})

### A:heavy_water + spa +lawnxNDVI+bathroom + above_one_acerage + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI+bathroom + above_one_acerage
result_31 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_31_2 = minimize(likelihood_f_jitted, result_31.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_31_3 = minimize(likelihood_f_jitted, result_31_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_31_4 = minimize(likelihood_f_jitted, result_31_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

### A:heavy_water_spa +bathroom + bedroom + lawnxNDVI + above_one_acerage + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + above_one_acerage + drought_stage
result_32 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_32_2 = minimize(likelihood_f_jitted, result_32.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_32_3 = minimize(likelihood_f_jitted, result_32_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_32_4 = minimize(likelihood_f_jitted, result_32_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

### A:heavy_water_app +spa + bedroom + lawnxNDVI + above_one_acerage + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + above_one_acerage + drought_stage
result_33 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_33_2 = minimize(likelihood_f_jitted, result_33.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_33_3 = minimize(likelihood_f_jitted, result_33_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_33_4 = minimize(likelihood_f_jitted, result_33_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

#so far result_33 is the best result

### A:heavy_water_app +spa + bathroom + bedroom + lawnxNDVI + above_one_acerage + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + bathroom + above_one_acerage 
result_34 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_34_2 = minimize(likelihood_f_jitted, result_34.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_34_3 = minimize(likelihood_f_jitted, result_34_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_34_4 = minimize(likelihood_f_jitted, result_34_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

### A:heavy_water_app +spa + bathroom + lawnxNDVI + above_one_acerage + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + bathroom + above_one_acerage + drought_stage
result_35 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_35_2 = minimize(likelihood_f_jitted, result_35.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_35_3 = minimize(likelihood_f_jitted, result_35_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_35_4 = minimize(likelihood_f_jitted, result_35_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

# NDVI scales to 0 ~ 1
### A:heavy_water_app +spa + bathroom + lawnxNDVI + above_one_acerage + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + bathroom 

result_36 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_36_2 = minimize(likelihood_f_jitted, result_36.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_36_3 = minimize(likelihood_f_jitted, result_36_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_36_4 = minimize(likelihood_f_jitted, result_36_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})


# NDVI scales to 0 ~ 1, use prev NDVI instead of prev NDVI diff, NDVI has being proportionized with month
### A:heavy_water_app +spa + bathroom + lawnxNDVI + above_one_acerage + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + bathroom +above_one_acerage
result_37 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_37_2 = minimize(likelihood_f_jitted, result_37.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_37_3 = minimize(likelihood_f_jitted, result_37_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_37_4 = minimize(likelihood_f_jitted, result_37_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})


# NDVI scales to 0 ~ 1, use prev NDVI instead of prev NDVI diff, NDVI has being proportionized with month
### A:heavy_water_app +spa + bathroom + lawnxNDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + bathroom 
result_38 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_38_2 = minimize(likelihood_f_jitted, result_38.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_38_3 = minimize(likelihood_f_jitted, result_38_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_38_4 = minimize(likelihood_f_jitted, result_38_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

## 38_4 is also a good result


# NDVI scales to 0 ~ 1, use prev NDVI diff, NDVI has being proportionized with month
### A:heavy_water_app +spa + bathroom + lawnxNDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + bathroom 
result_39 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_39_2 = minimize(likelihood_f_jitted, result_39.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_39_3 = minimize(likelihood_f_jitted, result_39_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_39_4 = minimize(likelihood_f_jitted, result_39_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})


# NDVI scales to 0 ~ 1, use prev NDVI, NDVI has being proportionized with month
# filtered out outlier bathroom and bedroom
### A:heavy_water_app +spa + bathroom + +bedroom + lawnxNDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + bathroom +  bedroom + drought_stage
result_40 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_40_2 = minimize(likelihood_f_jitted, result_40.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_40_3 = minimize(likelihood_f_jitted, result_40_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_40_4 = minimize(likelihood_f_jitted, result_40_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

# NDVI scales to 0 ~ 1, use prev NDVI, NDVI has being proportionized with month
# filtered out outlier bathroom
### A:heavy_water_app +spa + bathroom + lawnxNDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + bathroom 
result_41 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_41_2 = minimize(likelihood_f_jitted, result_41.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_41_3 = minimize(likelihood_f_jitted, result_41_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_41_4 = minimize(likelihood_f_jitted, result_41_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

# NDVI scales to 0 ~ 1, use prev NDVI, NDVI has being proportionized with month
# filtered out outlier bathroom
### A:heavy_water_app +spa + bathroom + lawnxNDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_app + spa +lawnxNDVI + bathroom 
result_42 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_42_2 = minimize(likelihood_f_jitted, result_42.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_42_3 = minimize(likelihood_f_jitted, result_42_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_42_4 = minimize(likelihood_f_jitted, result_42_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

# NDVI scales to 0 ~ 1, use prev NDVI _ diff, NDVI has being proportionized with month
# filtered out outlier bathroom
### A:heavy_water_app +spa + bathroom + lawnxNDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_app + spa +lawnxNDVI + bathroom 
result_43 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_43_2 = minimize(likelihood_f_jitted, result_43.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_43_3 = minimize(likelihood_f_jitted, result_43_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_43_4 = minimize(likelihood_f_jitted, result_43_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

# NDVI scales to 0 ~ 1, use prev NDVI, NDVI has being proportionized with month
# filtered out outlier bathroom
### A:heavy_water_spa + bathroom + lawnxNDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa +lawnxNDVI + bathroom 
result_44 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_44_2 = minimize(likelihood_f_jitted, result_44.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_44_3 = minimize(likelihood_f_jitted, result_44_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_44_4 = minimize(likelihood_f_jitted, result_44_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})


# NDVI scales to 0 ~ 1, use prev NDVI, NDVI has being proportionized with month
## Used sign boundaries 
### A:heavy_water_app +spa + bathroom + lawnxNDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + bathroom 
result_39_b = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_39_b_2 = minimize(likelihood_f_jitted, result_39_b.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_39_b_3 = minimize(likelihood_f_jitted, result_39_b_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_39_b_4 = minimize(likelihood_f_jitted, result_39_b_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})


# NDVI scales to 0 ~ 1, use prev NDVI, NDVI has being proportionized with month
## Used sign boundaries 
### A:heavy_water_app +spa + bedroombathroom + lawnxNDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + bedroombathroom 
result_39_c = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_39_c_2 = minimize(likelihood_f_jitted, result_39_c.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_39_c_3 = minimize(likelihood_f_jitted, result_39_c_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_39_c_4 = minimize(likelihood_f_jitted, result_39_c_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

# NDVI scales to 0 ~ 1, use prev NDVI, NDVI has being proportionized with month
# filtered out outlier bathroom
### A:heavy_water_app + spa + hvac_residential + lawnxNDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa +lawnxNDVI + hvac_residential 
result_45 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_45_2 = minimize(likelihood_f_jitted, result_45.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_45_3 = minimize(likelihood_f_jitted, result_45_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_45_4 = minimize(likelihood_f_jitted, result_45_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

# NDVI scales to 0 ~ 1, use prev NDVI, NDVI has being proportionized with month
# filtered out outlier bathroom
### A:heavy_water_app + spa + hvac_residential + lawnxNDVI + drought_stage + above_one_acre
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa +lawnxNDVI + hvac_residential 
result_46 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_46_2 = minimize(likelihood_f_jitted, result_46.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_46_3 = minimize(likelihood_f_jitted, result_46_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_46_4 = minimize(likelihood_f_jitted, result_46_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

# NDVI scales to 0 ~ 1, use prev NDVI, NDVI has being proportionized with month
# filtered out outlier bathroom
### A:heavy_water_app + spa + bathroom + lawnxNDVI + drought_stage + above_one_acre
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa +bathroom + lawnxNDVI 
result_47 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_47_2 = minimize(likelihood_f_jitted, result_47.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_47_3 = minimize(likelihood_f_jitted, result_47_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_47_4 = minimize(likelihood_f_jitted, result_47_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

# NDVI scales to 0 ~ 1, use prev NDVI, NDVI has being proportionized with month
# filtered out outlier bathroom
### A:heavy_water_app + spa + bathroom + lawnarea + prev_NDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa +bathroom + prev_NDVI
result_48 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_48_2 = minimize(likelihood_f_jitted, result_48.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_48_3 = minimize(likelihood_f_jitted, result_48_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_48_4 = minimize(likelihood_f_jitted, result_48_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

# filtered out outlier bathroom
### A:heavy_water_app + spa + bathroom + prev_NDVI + drought_stage + above_one_acre
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa +bathroom + prev_NDVI
result_49 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_49_2 = minimize(likelihood_f_jitted, result_49.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_49_3 = minimize(likelihood_f_jitted, result_49_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_49_4 = minimize(likelihood_f_jitted, result_49_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})
 
# filtered out outlier bathroom
### A:heavy_water_spa + bathroom + prev_NDVI + drought_stage + above_one_acre
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa +bathroom + prev_NDVI
result_50 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_50_2 = minimize(likelihood_f_jitted, result_50.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_50_3 = minimize(likelihood_f_jitted, result_50_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_50_4 = minimize(likelihood_f_jitted, result_50_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

# NDVI scales to 0 ~ 1, use prev NDVI diff, NDVI has being proportionized with month
# filtered out outlier bathroom
### A:heavy_water_app +spa + bedroom + lawnxNDVI + above_one_acerage + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + above_one_acerage + drought_stage
result_51 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_51_2 = minimize(likelihood_f_jitted, result_51.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_51_3 = minimize(likelihood_f_jitted, result_51_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_51_4 = minimize(likelihood_f_jitted, result_51_3.x, method = 'Nelder-Mead', options={'maxfev': 20000}) 

# NDVI scales to 0 ~ 1, use prev NDVI diff in sqft, NDVI has being proportionized with month
# filtered out outlier bathroom
### A:heavy_water_app +spa + bedroom + lawnxNDVI + above_one_acerage + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + above_one_acerage + drought_stage
result_52 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_52_2 = minimize(likelihood_f_jitted, result_52.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_52_3 = minimize(likelihood_f_jitted, result_52_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_52_4 = minimize(likelihood_f_jitted, result_52_3.x, method = 'Nelder-Mead', options={'maxfev': 20000}) 


# NDVI scales to 0 ~ 1, use prev NDVI in sqft, NDVI has being proportionized with month
# filtered out outlier bathroom
### A:heavy_water_spa + bathroombedroom + lawnxNDVI + above_one_acerage + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + above_one_acerage + drought_stage
result_53 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_53_2 = minimize(likelihood_f_jitted, result_53.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_53_3 = minimize(likelihood_f_jitted, result_53_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_53_4 = minimize(likelihood_f_jitted, result_53_3.x, method = 'Nelder-Mead', options={'maxfev': 20000}) 

# NDVI scales to 0 ~ 1, use prev NDVI in acre, NDVI has being proportionized with month
# filtered out outlier bathroom
### A:heavy_water_spa + bathroom + lawnxNDVI  + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+bathroom + lawnxNDVI  + drought_stage
result_54 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_54_2 = minimize(likelihood_f_jitted, result_54.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_54_3 = minimize(likelihood_f_jitted, result_54_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_54_4 = minimize(likelihood_f_jitted, result_54_3.x, method = 'Nelder-Mead', options={'maxfev': 20000}) 

# NDVI scales to 0 ~ 1, use prev NDVI in acre, NDVI has being proportionized with month
# filtered out outlier bathroom
### A:heavy_water_spa + bathroom + lawnxNDVI  + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A: no A, only constant for alpha and rho
result_55 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_55_2 = minimize(likelihood_f_jitted, result_55.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_55_3 = minimize(likelihood_f_jitted, result_55_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_55_4 = minimize(likelihood_f_jitted, result_55_3.x, method = 'Nelder-Mead', options={'maxfev': 20000}) 


# NDVI scales to 0 ~ 1, use prev NDVI in acre, NDVI has being proportionized with month
# filtered out outlier bathroom
### A:spa + bathroom + lawnxNDVI  + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A: spa + bathroom + lawnxNDVI  + drought_stage
result_56 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_56_2 = minimize(likelihood_f_jitted, result_56.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_56_3 = minimize(likelihood_f_jitted, result_56_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_56_4 = minimize(likelihood_f_jitted, result_56_3.x, method = 'Nelder-Mead', options={'maxfev': 20000}) 

# NDVI scales to 0 ~ 1, use prev NDVI in acre, NDVI has being proportionized with month
# filtered out outlier bathroom
### A:heavy_water_app+ spa + bathroom + NDVI + above_one_acerage  + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A: heavy_water_spa + bathroom + NDVI + above_one_acerage  + drought_stage
result_57 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_57_2 = minimize(likelihood_f_jitted, result_57.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_57_3 = minimize(likelihood_f_jitted, result_57_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_57_4 = minimize(likelihood_f_jitted, result_57_3.x, method = 'Nelder-Mead', options={'maxfev': 20000}) 


# NDVI scales to 0 ~ 1, use prev NDVI instead of prev NDVI diff, NDVI has being proportionized with month
### A:heavy_water_app +spa + bathroom + lawnxNDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + bathroom 
result_38_b = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_38_b_2 = minimize(likelihood_f_jitted, result_38_b.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_38_b_3 = minimize(likelihood_f_jitted, result_38_b_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_38_b_4 = minimize(likelihood_f_jitted, result_38_b_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

# NDVI scales to 0 ~ 1, use prev NDVI instead of prev NDVI diff, NDVI has being proportionized with month
### A:heavy_water_app +spa + bedroom + lawnxNDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + bedroom
result_38_c = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_38_c_2 = minimize(likelihood_f_jitted, result_38_c.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_38_c_3 = minimize(likelihood_f_jitted, result_38_c_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_38_c_4 = minimize(likelihood_f_jitted, result_38_c_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

# NDVI scales to 0 ~ 1, use prev NDVI instead of prev NDVI diff, NDVI has being proportionized with month
### A:heavy_water_spa + bedroom + lawnxNDVI + above_one_acre + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + bedroom
result_38_d = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_38_d_2 = minimize(likelihood_f_jitted, result_38_d.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_38_d_3 = minimize(likelihood_f_jitted, result_38_d_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_38_d_4 = minimize(likelihood_f_jitted, result_38_d_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

# NDVI scales to 0 ~ 1, use prev NDVI instead of prev NDVI diff, NDVI has being proportionized with month
### A:heavy_water_spa + bedroom + lawnxNDVI + above_one_acre + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+lawnxNDVI + bedroom
### Sign bound all parameters
result_38_e = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_38_e_2 = minimize(likelihood_f_jitted, result_38_e.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_38_e_3 = minimize(likelihood_f_jitted, result_38_e_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_38_e_4 = minimize(likelihood_f_jitted, result_38_e_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})


##### Version 3: Using more refined NDVI (10m by 10m), Goal: need to make sure SE is not too small, otherwise could be overidentified. 

# NDVI from -1 ~ 1, use prev NDVI, NDVI has being proportionized with month
### A:heavy_water_spa + bedroom + NDVI + above_one_acre + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: heavy_water_spa+bedroom+NDVI
### Sign bound all parameters

result_58 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_58_2 = minimize(likelihood_f_jitted, result_58.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_58_3 = minimize(likelihood_f_jitted, result_58_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_58_4 = minimize(likelihood_f_jitted, result_58_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

## The coefficients are fine, but SE are too small, 
### A:heavy_water_spa + bedroom + NDVI + drought_stage
## Z: Tmax + Percipitation + IQR
## A:alpha: bathroom+NDVI_total_PRCP
result_59 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_59_2 = minimize(likelihood_f_jitted, result_59.x, method = 'Nelder-Mead', options={'maxfev': 20000})

## SE got slightly bigger by delete a couple of parameters
## Use VIF small columns (bedroom instead of bathroom)
### A:heavy_water_spa + bedroom + NDVI + drought_stage
## Z: Tmax + Percipitation + IQR
## A:alpha: bedroom+ NDVI
result_60 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_60_2 = minimize(likelihood_f_jitted, result_60.x, method = 'Nelder-Mead', options={'maxfev': 20000})

## SE not as good as 59, but 60_2 SE is huge for equation inside alpha. 
## Use VIF small columns (bedroom instead of bathroom)
### A:heavy_water_spa + bedroom + NDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:just do alpha and rho
result_61 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_61_2 = minimize(likelihood_f_jitted, result_61.x, method = 'Nelder-Mead', options={'maxfev': 20000})


## SE not as good as 59, getting back to 59
## Use VIF small columns (bedroom instead of bathroom)
### A:heavy_water_spa + bedroom + NDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:heavy_water_app + NDVI + total_PRCP
### Everything start at 0
result_62 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_62_2 = minimize(likelihood_f_jitted, result_62.x, method = 'Nelder-Mead', options={'maxfev': 20000})

## SE not as good as 59, getting back to 59
## Use VIF small columns (bedroom instead of bathroom)
### A:heavy_water_spa + bedroom + NDVI + drought_stage
## Z:  Percipitation + IQR
## A:heavy_water_app + NDVI + total_PRCP
### Everything start at 0
result_63 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
#result_63_2 = minimize(likelihood_f_jitted, result_63.x, method = 'Nelder-Mead', options={'maxfev': 20000})

## Back to 59_2
### A:heavy_water_spa + bedroom + NDVI + drought_stage
## Z: Tmax + Percipitation + IQR
## A:alpha: bathroom+NDVI_total_PRCP
## Fix d_k for CAP
result_59_b = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_59_b_2 = minimize(likelihood_f_jitted, result_59_b.x, method = 'Nelder-Mead', options={'maxfev': 20000})

### A:heavy_water_app + spa + bedroom + NDVI + drought_stage
## Z: Tmax + Percipitation + IQR
## A:alpha: bathroom + heavy_water_app+ NDVI
result_59_c = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_59_c_2 = minimize(likelihood_f_jitted, result_59_c.x, method = 'Nelder-Mead', options={'maxfev': 20000})

### A:heavy_water_spa + bedroom + NDVI + drought_stage
## Z: Tmax + Percipitation + IQR
## A:alpha: bathroom + NDVI
result_59_d = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_59_d_2 = minimize(likelihood_f_jitted, result_59_d.x, method = 'Nelder-Mead', options={'maxfev': 20000})

### A:heavy_water_spa + bedroom + NDVI + drought_stage
## Z: Tmax + Percipitation + IQR
## A:alpha: bathroom + heavy_water_app
result_59_e = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_59_e_2 = minimize(likelihood_f_jitted, result_59_e.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_59_e_3 = minimize(likelihood_f_jitted, result_59_e_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_59_e_4 = minimize(likelihood_f_jitted, result_59_e_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

### A:heavy_water_spa + bedroom + NDVI + drought_stage
## Z: Tmax + Percipitation + IQR
## A:alpha: bathroom
result_59_f = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_59_f_2 = minimize(likelihood_f_jitted, result_59_f.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_59_f_3 = minimize(likelihood_f_jitted, result_59_f_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_59_f_4 = minimize(likelihood_f_jitted, result_59_f_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

### A: bedroom + NDVI + drought_stage
## Z: Tmax + Percipitation + IQR
## A:alpha: heavy_water_spa
result_59_g = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
result_59_g_2 = minimize(likelihood_f_jitted, result_59_g.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_59_g_3 = minimize(likelihood_f_jitted, result_59_g_2.x, method = 'Nelder-Mead', options={'maxfev': 20000})
result_59_g_4 = minimize(likelihood_f_jitted, result_59_g_3.x, method = 'Nelder-Mead', options={'maxfev': 20000})

### A: heavy_water_spa + NDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: bedroom
result_59_h = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### A: heavy_water_spa + NDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: NDVI
result_59_i = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### A: heavy_water_spa + NDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: NDVI + CAP_HH
result_59_j = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### A: heavy_water_spa + NDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: bedroom + CAP_HH
result_59_k = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})


### A: heavy_water_spa + NDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: bedroom +NDVI +  CAP_HH
result_59_l = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### A: heavy_water_spa + NDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: bathroom +NDVI +  CAP_HH
result_59_m = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### A: heavy_water_spa + NDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: bathroom +NDVI +  CAP_HH, just rho
result_60 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### A: heavy_water_spa + NDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: bedroom + CAP_HH, just rho
result_61 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})


### A: heavy_water_spa + NDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## A:alpha: bedroom + NDVI, just rho
result_62 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

#### Starting with just alpha and rho, then expand the alpha function
### A: heavy_water_spa + NDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## just alpha and just rho
result_63 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

#### The Alpha is too small, start with this format first 
### A: heavy_water_spa + bedroom + NDVI + drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## just alpha and just rho
result_64 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

#### The Alpha is too small, start with this format first 
### A: heavy_water_spa + bedroom + NDVI + above_one_acre+ drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## just alpha and just rho, use jnp.exp(alpha) and jnp.exp(rho)
result_65 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

#### The Alpha is too small, start with this format first 
### A: heavy_water_spa + bedroom + lawn area x NDVI + above_one_acre+ drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## just alpha and just rho, use jnp.exp(alpha) and jnp.exp(rho)
result_66 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

#### 66 is not too bad, maybe need all 5 variables  and se is a tiny bit better
### A: heavy_water_spa_area + bedroom + lawn area x NDVI + above_one_acre+ drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## just alpha and just rho, use jnp.exp(alpha) and jnp.exp(rho)
result_67 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

#### 67 is also not too bad, ned all 5 for A, and use exp instead of abs
### A: heavy_water_spa_area + bathroom + lawn area x NDVI + above_one_acre+ drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## just alpha and just rho, use jnp.exp(alpha) and jnp.exp(rho)
result_68 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

#### 68 is better, need all 5 for A, and use exp instead of abs, but rho is super small
### A: heavy_water + spa + bathroom + lawn area x NDVI + above_one_acre+ drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## just alpha and just rho, use jnp.exp(alpha) and jnp.exp(rho)
result_69 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 69 is the best so far
### A: heavy_water + spa + bathroom + lawn area x NDVI + above_one_acre+ drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## add alpha A: heavy_water_spa + bedroom + lawnarea x NDVI
result_70 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### In 70, adding detailed A for alpha does not improve alpha, but still not bad, rho is too small. 
### A: heavy_water + spa + bathroom + NDVI + above_one_acre+ drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## add alpha A: heavy_water_spa + bedroom + lawnarea x NDVI
result_71 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### In 71, alpha = 0.004, rho too small
### A: heavy_water + spa + bathroom + NDVI + above_one_acre+ drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## add alpha A: heavy_water_spa + bedroom + NDVI
result_72 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### In 72, alpha is way too small, rho is a bit better, need to use lawn_arexNDVI
### A: heavy_water_spa + bathroom + NDVI + above_one_acre+ drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## add alpha A: heavy_water_spa + bedroom + lawn_areaxNDVI + Tmax + Prcp, rho A: heavy_water_spa + bedrom + prevNDVI
#### This is exactly like Wang and Wolak 2022
result_73 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### In 73 both alpha and rho are way too small, SE for price and income regressions are too big, especially income
### The lawnareaxNDVI in price has huge SE
### A: heavy_water_spa + bathroom + NDVI + above_one_acre+ drought_stage
## Z: Tmax + IQR + Percipitation + IQR
## Also delete heavy water app in A_alpha and A_rho since wang and wolak does not have it. 
## add alpha A:  bedroom + prev_NDVI + Tmax + Prcp, rho A: bedrom + prevNDVI
result_74 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### In alpha A and alpha rho, should only use NDVI, 

### result_74 is the best so far, need to fix the income 
### A income = heavy_water_spa + bedroom
### mean alpha 0.12, median 0.00074
result_75 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### result_75 fix the income, but alpha becomes small, 
### A income = heavy_water_spa + prev_NDVI
result_76 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 76 boost the alpha (a bit too much), and rho now is too small, adding NDVI makes rho just so small
### Need to keep fixing income
### A alpha heavy_water_spa + prev_NDVI + Tmax + PRCP , A income = heavy_water_spa + prev_NDVI
### mean alpha = 0.8
result_77 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
'''
### 77 is the best so far, SE is still on the small end, but the sturcture will be like this. 
### A alpha heavy_water_spa + prev_NDVI + Tmax + PRCP , A income = heavy_water_spa +prev_NDVI

### 77 is the best so far, SE is still on the small end, but the sturcture will be like this. 
### alpha seems a bit too big
### Adding bedrom to A alpha? since in 72 73, including bedroom in 72 73 makes alpha smaller
### A alpha heavy_water_spa +bedroom+ prev_NDVI + Tmax + PRCP , A income = heavy_water_spa +prev_NDVI
result_78 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

"""
#### 78 is the best so far, better than 77
#### mean alpha = 0.38, the rho is a tad bit too small. 
## A heavy_water_spa +bedroom+ prev_NDVI + above_one_acre+drought_stage
### A alpha heavy_water_spa +bedroom+ prev_NDVI + Tmax + PRCP , A income = heavy_water_spa +bedroom+prev_NDVI
result_79 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### rho becomes better, but alpha becomes too small
## A heavy_water_spa +bedroom+ prev_NDVI + above_one_acre+drought_stage
### A alpha heavy_water_spa +bathroom+ prev_NDVI + Tmax + PRCP , A income = heavy_water_spa+prev_NDVI
result_80 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 80 is not bad, alpha is a bit too small, se is too small
## A heavy_water_spa +bathroom+ prev_NDVI + above_one_acre+drought_stage
### A alpha heavy_water_spa +bathroom+ prev_NDVI + Tmax + PRCP , A income = heavy_water_spa+prev_NDVI
result_80_b = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 80b improves on se and range of alpha, but alpha is still too small.
## A heavy_water_spa +bathroom+ prev_NDVI + above_one_acre+drought_stage
### A alpha heavy_water_spa +bedroom+ prev_NDVI + Tmax + PRCP , A income = heavy_water_spa+prev_NDVI
result_80_c = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 80c both alpha and rho are too small, rho is way off
## A heavy_water_spa +bedroom+ prev_NDVI + above_one_acre+drought_stage
### A alpha heavy_water_spa +bedroom+ prev_NDVI + Tmax + PRCP , A income = heavy_water_app +prev_NDVI
result_78_b = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

###78b alpha is good, but rho is way too small
## A heavy_water_spa +bedroom+ prev_NDVI + above_one_acre+drought_stage
### A alpha heavy_water_app +bedroom+ prev_NDVI + Tmax + PRCP , A income = heavy_water_app +prev_NDVI
result_78_c = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

###78c is not good
## A heavy_water_spa +bedroom+ prev_NDVI + above_one_acre+drought_stage
### A alpha heavy_water_spa +bedroom+ prev_NDVI + Tmax + PRCP , A income = spa +prev_NDVI
result_78_d = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 78 d is not bad, the only thing is SE is too small
## A heavy_water_spa +bedroom+ prev_NDVI + above_one_acre+drought_stage
### A alpha heavy_water_spa +bedroom+ prev_NDVI + Tmax + PRCP , A income = heavy_water_spa +prev_NDVI
result_78_e = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

#### 78 is the best so far, better than 77
## A heavy_water_spa +bedroom+ prev_NDVI + above_one_acre+drought_stage
### A alpha heavy_water_spa +bedroom+ prev_NDVI + Tmax + PRCP , A income = heavy_water_spa +above_one_acre+prev_NDVI
result_79_b = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

#### 79b reduce alpha and rho isn't big either
## A heavy_water_spa +bedroom+ prev_NDVI + above_one_acre+drought_stage
### A alpha heavy_water_spa +bedroom+ prev_NDVI + Tmax + PRCP , A income = heavy_water_spa +bedroom
result_79_c = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 79c is quite bad, se is ok.
## A heavy_water_spa +bedroom+ prev_NDVI + above_one_acre+drought_stage
### A alpha heavy_water_spa +bedroom+ prev_NDVI + Tmax + PRCP , A income = heavy_water_spa +bedroom+prev_NDVI
result_79 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 79 both alpha and rho are small, se for income is big
## A heavy_water_spa +bedroom+ prev_NDVI + above_one_acre+drought_stage
### A alpha heavy_water_spa +bedroom+ prev_NDVI + Tmax + PRCP , A income = heavy_water_spa +bathroom+prev_NDVI
result_79_b = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 79 both alpha and rho are small, se for income is big
## A heavy_water_spa +bedroom+ prev_NDVI + above_one_acre+drought_stage
### A alpha heavy_water_spa +bedroom+ prev_NDVI, A income = heavy_water_spa +bedroom+prev_NDVI
result_81 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 81, both rho and alpha are too small, se is nice
## A heavy_water_spa +bedroom+ prev_NDVI + above_one_acre+drought_stage
### A alpha heavy_water_spa +bedroom+ prev_NDVI + Tmax + PRCP, A income = heavy_water_spa +bedroom+prev_NDVI+ Tmax + PRCP
result_82 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 82, rho is better, but alpha now too small and se big
## A heavy_water_spa +bedroom+ prev_NDVI + above_one_acre+drought_stage
### A alpha heavy_water_spa +bedroom+ prev_NDVI + Tmax + PRCP, A income = heavy_water_spa + Tmax + PRCP
result_83 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 83, rho is better, but alpha now too small and se big
## A heavy_water_spa +bedroom+ prev_NDVI + above_one_acre+drought_stage
### A alpha heavy_water_spa +bedroom+ prev_NDVI + Tmax + PRCP, A income = heavy_water_spa + Prcp
result_84 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 84, rho is better, but alpha now too small and se big
## A heavy_water_spa +bedroom+ prev_NDVI + above_one_acre+drought_stage
### A alpha heavy_water_app +bathroom+ prev_NDVI + Tmax + PRCP, A income = heavy_water_spa + Prcp
result_85 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
"""
### 77 is the best so far, SE is still on the small end, but the sturcture will be like this. 
### A alpha heavy_water_spa + prev_NDVI + Tmax + PRCP , A income = heavy_water_spa +prev_NDVI
result_77 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
### alpha mean = 0.11
### alpha median = 0.013
### rho mean = 0.244
### rho median = 0.2568
"""
### A alpha bedroom + prev_NDVI + Tmax + PRCP , A income = heavy_water_spa +prev_NDVI
result_86 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 86 makes alpha too extreme, and se is a bit too small, 86 and 77 are very similar
### A alpha heavy_water_spa + bedroom + prev_NDVI + Tmax + PRCP , A income = prev_NDVI
result_87 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 87 is pretty bad
### A alpha heavy_water_spa + bedroom + prev_NDVI + Tmax + PRCP , A income = heavy_water_app
result_88 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})


########################## Think alpha and rho equation as interaction terms ############################
### A alpha heavy_water_spa +bedroom+ prev_NDVI + Tmax + PRCP , A income = heavy_water_spa +prev_NDVI
### no exp on alpha and rho, easier to interpret
result_78_b = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### A alpha heavy_water_spa +bedroom+ prev_NDVI + Tmax + PRCP , A income = heavy_water_spa +bedroom + prev_NDVI
### no exp on alpha and rho, easier to interpret
result_79_b = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### Both 78b and 79b are good but se pretty smalls

### A alpha bedroom + prev_NDVI + Tmax + PRCP , A income = bedroom  + prev_NDVI
### no exp on alpha and rho, easier to interpret
result_77_c = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### A alpha heavy_water_spa + prev_NDVI + Tmax + PRCP , A income = heavy_water_spa  + prev_NDVI
### no exp on alpha and rho, easier to interpret
result_77_b = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### less variable seems to make alpha big, and rho small, se is still small. 
### Overfitting?
result_77_d = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 77d has nice results but se is still too small
### less extreme values
### Try delete parameters
### A alpha heavy_water_spa+ bedroom , A income = heavy_water_spa+ bedroom 
result_89 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### se is still quite small, and alpha is too small
### A alpha bedroom + NDVI, A income = bedroom 
result_90 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### se is still quite small, and alpha is too small
### A heavy_water_spa +NDVI + above_one_acre + drought_stage
### A alpha bedroom + NDVI, A income = bedroom 
result_91 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### Need to add complex model for A Alpha? 
### A heavy_water_spa +NDVI + drought_stage
### A alpha bedroom + NDVI + mean_Tmax + total_Prcp, A income = bedroom 
result_92 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### Need to add complex model for A Alpha? 
### A heavy_water_spa +bathroom + NDVI +
### A alpha bedroom + mean_Tmax + total_Prcp, A income = bedroom 
result_93 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### Need to add complex model for A Alpha? 
### A bathroom + NDVI
### A alpha bathroom + mean_Tmax + total_Prcp, A income = bathroom
result_94 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### se is still too small, make it less complex?
### A bathroom + NDVI
### A alpha bathroom + total_Prcp, A income = bathroom
result_95 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

"""
### The se is till quite small, alpha is pretty big as well. 
### try exp on A alpha, but not on A rho
### A bathroom + NDVI
### A alpha heavy_water_spa +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + NDVI
result_96 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### Certainly in the right direction
### A bathroom + NDVI + drought_stage
### A alpha heavy_water_spa +bedroom + NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + bedroom + NDVI
result_97 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### Definitly right direction, potential overfitting
### A bathroom + NDVI + drought_stage
### A alpha bedroom + NDVI + mean_tmax +  total_Prcp, A income = bedroom + NDVI
result_98 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### not in the right direction
### A bathroom + NDVI + drought_stage
### A alpha heavy_water_spa + NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + NDVI
result_98_b = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### wrong direction
### A bathroom + NDVI + drought_stage
### A alpha heavy_water_spa +bedroom + NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + NDVI
result_99 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### se bigger once A alpha more complex, but alpha becomes smaller
### A bedroom + NDVI + above_one_acre+ drought_stage
### A alpha heavy_water_spa +bedroom + NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + NDVI
result_100 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### A bathroom + NDVI
### A alpha heavy_water_spa +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + NDVI
result_96 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### A bathroom + NDVI
### A alpha bedroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + NDVI
result_96_b = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
#### se mean 0.02
### alpha mean 0.0109
### alpha median 0.0108
### rho mean 0.05

### 96b is the best so far, need to increase alpha
### A bathroom + NDVI
### A alpha heavy_water_spa+ bedroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + NDVI
result_96_c = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### Not a lot of changes
### A bathroom + NDVI
### A alpha heavy_water_spa+ bathroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + NDVI
result_96_d = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### bathroom decrease se
### A bathroom + NDVI
### A alpha heavy_water_app+ bedroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + NDVI
result_96_e = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### heavy water app decrease se
### A bathroom + NDVI
### A alpha spa+ bedroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + NDVI
result_96_f = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### just spa is terrible
### A bedroom + NDVI
### A alpha heavy_water_spa+ bedroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + NDVI
result_96_g = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})


### retry from 96b
### A bathroom + NDVI
### A alpha bathroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + NDVI
result_96_h = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
#### se mean 0.004
### alpha mean 0.0545
### alpha median 0.0530
### rho mean 0.03

### A bathroom + NDVI
### A alpha bathroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_app + NDVI
result_96_i = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### Change from heavy water spa to heavy water app make it se small and alpha too big
### A bathroom + NDVI
### A alpha bathroom +NDVI + mean_tmax +  total_Prcp, A income = spa + NDVI
result_96_j = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
#### se mean 0.1458
### alpha mean 0.0014
### alpha median 0.0014
### rho mean 0.17

### spa in A income make se bigger, but alpha is too small. 
### A bathroom + NDVI
### A alpha bathroom +NDVI + mean_tmax +  total_Prcp, A income = bathroom + NDVI
result_96_k = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
### se is too big

### A bathroom + NDVI
### A alpha bathroom +NDVI + mean_tmax +  total_Prcp, A income = bedroom + NDVI
result_96_l = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
### se is too small

### A bathroom + NDVI
### A alpha bathroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + bedroom + NDVI
result_96_m = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
#### se mean 0.02
### alpha mean 0.007
### alpha median 0.003
### rho mean 0.19

##########################################################################################################################
### A bathroom + NDVI
### A alpha bathroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_app + bedroom + NDVI
result_96_n = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
#### se mean 0.0019
### alpha mean 0.14765
### alpha median 0.1295
### rho mean 0.15
##########################################################################################################################

### A bathroom + NDVI
### A alpha bathroom +NDVI + mean_tmax +  total_Prcp, A income = spa + bedroom + NDVI
result_96_o = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
#### se mean 0.01268
### alpha mean 0.01
### alpha median 0.012
### rho mean 0.004

### A bathroom + NDVI
### A alpha bathroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + bathroom + NDVI
result_96_m2 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
#### se mean 0.0006
### alpha mean 1.439
### alpha median 1.126
### rho mean 0.500

### A bathroom + NDVI
### A alpha bathroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_app + bathroom + NDVI
result_96_n2 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
#### se mean 0.00091277
### alpha mean 0.28
### alpha median 0.00699
### rho mean 0.38

### A bathroom + NDVI
### A alpha bathroom +NDVI + mean_tmax +  total_Prcp, A income = spa + bathroom + NDVI
result_96_o2 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
#### se mean 0.0007
### alpha mean 1.38
### alpha median 1.33
### rho mean 0.75

### A bathroom + NDVI
### A alpha bedroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + bedroom + NDVI
result_96_m3 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
#### se mean 0.004
### alpha mean 0.04
### alpha median 0.04
### rho mean 0.33

##########################################################################################################################
### A bathroom + NDVI
### A alpha bedroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_app + bedroom + NDVI
result_96_n3 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
#### se mean 0.0012
### alpha mean 0.49
### alpha median 0.43
### rho mean 0.11
##########################################################################################################################

### A bathroom + NDVI
### A alpha bedroom +NDVI + mean_tmax +  total_Prcp, A income = spa + bedroom + NDVI
result_96_o3 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
#### se mean 0.00156
### alpha mean 0.007
### alpha median 3.00e-14
### rho mean 0.007

### A bathroom + NDVI
### A alpha bedroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + bathroom + NDVI
result_96_m4 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
### se mean 0.07
### alpha mean 0.00109
### alpha median 0.00014
### rho mean 0.59

### A bathroom + NDVI
### A alpha bedroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_app + bathroom + NDVI
result_96_n4 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
### se mean 0.00084
### alpha mean 0.12
### alpha median 0.0132
### rho mean 0.32

### A bathroom + NDVI
### A alpha bedroom +NDVI + mean_tmax +  total_Prcp, A income = spa + bathroom + NDVI
result_96_o4 = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
### This is pretty bad

### A bedroom + NDVI
### A alpha bathroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_app + bedroom + NDVI
result_96_nb = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
### se mean 0.00077
### alpha mean 0.59
### alpha median 0.169
### rho mean 0.61

### A bedroom + NDVI
### A alpha bedroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_app + bedroom + NDVI
result_96_n3b = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
### This is pretty bad

### A bathroom + NDVI
### A alpha heavy_water_app + bathroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_app + bedroom + NDVI
result_96_nc = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})
### se mean 0.00094
### alpha mean 0.17
### alpha median 0.0018
### rho mean 0.14

### A bathroom + NDVI
### A alpha heavy_water_app + bedroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_app + bedroom + NDVI
result_96_n3c = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### Overall n3 is a bit better

"""

### Change A to bedroom is also terrible, need to add drought_stage
### A bathroom + NDVI + drought_stage
### A alpha bedroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + NDVI
result_99_b = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### from 99 to 99b, remove heavy water spa, didn't change anything, se big, alpha small
### A bathroom + NDVI + drought_stage
### A alpha heavy_water_spa+bathroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + NDVI
result_99_c = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### changing bedroom to bathroom doesn't seem to change anything
### A bedroom + NDVI + drought_stage
### A alpha heavy_water_spa+bedroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + NDVI
result_99_d = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 99 d has good alpha and rho, but se too small, heavy_water_app decreases se, so try just spa
### A bedroom + NDVI + drought_stage
### A alpha spa+bedroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + NDVI
result_99_e = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### spa makes se big, try heavy water app
### A bedroom + NDVI + drought_stage
### A alpha heavy_water_app+bedroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + NDVI
result_99_f = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 99f is definitly better, try bathroom with heavy water app?
### A bedroom + NDVI + drought_stage
### A alpha heavy_water_app+bathroom +NDVI + mean_tmax +  total_Prcp, A income = heavy_water_spa + NDVI
result_99_g = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

#### bathroom makes se huge
### A bedroom + NDVI + drought_stage
### A alpha heavy_water_app+bedroom +NDVI +  total_Prcp, A income = heavy_water_spa + NDVI
result_99_h = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

#### bathroom makes se huge
### A bedroom + NDVI + drought_stage
### A alpha heavy_water_app+bedroom +mean_Tmax+ total_Prcp, A income = heavy_water_spa + NDVI
result_99_i = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

#### NDVI has to be included, se becomes so small if not included
### A bedroom + NDVI + drought_stage
### A alpha bedroom +NDVI + mean_Tmax+ total_Prcp, A income = heavy_water_spa + NDVI
result_99_j = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

#### heavy_water_app has to be included
### A bedroom + NDVI + drought_stage
### A alpha heavy_water_app+ bedroom +NDVI + mean_Tmax, A income = heavy_water_spa + NDVI
result_99_k = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

#### in this case, all 5 are needed
### A bedroom + NDVI + drought_stage
### A alpha heavy_water_app+ bedroom +NDVI + mean_Tmax + total_PRCP, A income = NDVI
result_99_l = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### Change A income does change alpha result, but delete heavy_water_spa makes se smaller and alpha smaller
### A bedroom + NDVI + drought_stage
### A alpha heavy_water_app+ bedroom +NDVI + mean_Tmax + total_PRCP, A income = heavy_water_app + NDVI
result_99_m = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### heavy_water_app in A income does not work
### A bedroom + NDVI + drought_stage
### A alpha heavy_water_app+ bedroom +NDVI + mean_Tmax + total_PRCP, A income = spa + NDVI
result_99_n = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 99n is the best so far
### A bedroom + NDVI + drought_stage
### A alpha heavy_water_spa + bedroom +NDVI + mean_Tmax + total_PRCP, A income = spa + NDVI
result_99_o = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 99o has small se and small alpha
### A bedroom + NDVI + drought_stage
### A alpha spa + bedroom +NDVI + mean_Tmax + total_PRCP, A income = spa + NDVI
result_99_p = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 99o has small se and small alpha
### A bedroom + NDVI + drought_stage
### A alpha heavy_water_app + bathroom +NDVI + mean_Tmax + total_PRCP, A income = spa + NDVI
result_99_q = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})

### 99q has large se and small alpha
### A bathroom + NDVI + drought_stage
### A alpha heavy_water_app + bedroom +NDVI + mean_Tmax + total_PRCP, A income = spa + NDVI
result_99_r = minimize(likelihood_f_jitted, starting, method = 'Nelder-Mead', options={'maxfev': 20000})


"""



### Calculate Variance
 
def find_index(arr, target):
    indices = jnp.nonzero(arr == target)[0]
    if indices.size == 0:
        return -1  # Element not found
    return int(indices[0])  # Return the index as an integer

ste = 0.0001

r = jnp.array(result_96_n3.x)

#ll = jnp.array(likelihood_f_base_jitted(r))

#def change_beta (r_i, ste = ste, beta_l = r):
def change_beta (k, ste = ste, beta_l = r):
    l = len(beta_l)
    change_l = jnp.zeros(l)
    #k = jnp.where(beta_l == r_i)[0][0]
    #k = find_index(beta_l, r_i)
    #change_l[k] = ste
    change_l = change_l.at[k].set(ste)
    return_array = jnp.multiply(beta_l, change_l+1)
    return return_array

change_beta_jitted = jax.jit(change_beta)

#def calculate_prime (k, ste = ste, beta_l = r, base_ll = ll):
   # ll_changed = jnp.array(likelihood_f_base_jitted(change_beta_jitted(k)))
  #  prime = (ll_changed - base_ll) 
    #/ (ste * beta_l[k])
 #   return prime

#calculate_prime_jitted = jax.jit(calculate_prime)

#def calculate_var (k, ste = ste, beta_l = r):
 #   prime = calculate_prime_jitted(k)
  #  var = 1/ jnp.sum(jnp.square(prime))
   # return var

def calculate_var (k, ste = ste, beta_l = r):
    beta_l = jnp.array(beta_l)
    var = 1 / jnp.sum(jnp.square(( likelihood_f_base_jitted(change_beta_jitted(k)) -likelihood_f_base_jitted(beta_l)) / (ste*beta_l[k])))
    return var
    
vcalculate_var = jnp.vectorize(calculate_var)

var_l = vcalculate_var(np.arange(len(r)))
se_l = jnp.sqrt(var_l)

result = np.array([r, se_l])

beta = result[0]
se = result[1]

b1 = jnp.array([jnp.exp(beta[0]), jnp.exp(beta[1]),
                #-1*jnp.exp(beta[2]), 
                #-1*jnp.exp(beta[3])
                #-1*jnp.exp(beta[4])
                #-1*jnp.exp(beta[5])
                ])
b2 = jnp.array([beta[2], beta[3], -1*jnp.exp(beta[4]), beta[5]])
c_o = beta[6]
b4 = jnp.array([beta[7], beta[8], beta[9], beta[10]
                #, beta[11]
                ])
c_alpha = beta[11]
b6 = jnp.array([beta[12], beta[13], beta[14]])
c_rho = beta[15]
#alpha = jnp.exp(beta[11])
#rho = jnp.exp(beta[15])
sigma_eta = abs(beta[16])
sigma_nu = abs(beta[17])

alpha = abs(jnp.exp(
    jnp.dot(A_current_price, b4)
                    + c_alpha
                    )
                    )

rho = abs(
#jnp.exp(
jnp.dot(A_current_income, b6)
                    + c_rho
                    #)
                    )



filtered_alpha = alpha[np.isfinite(alpha)]
filtered_rho = rho[np.isfinite(rho)]

beta = jnp.concatenate([b1,  b2, jnp.array([c_o]), b4, jnp.array([c_alpha]), b6, jnp.array([c_rho]), jnp.array([sigma_eta]), jnp.array([sigma_nu]) ])

se[0] = se[0] * jnp.exp(beta[0])
se[1] = se[1] * jnp.exp(beta[1])
#se[2] = se[2] * jnp.exp(beta[2])
#se[3] = se[3] * jnp.exp(beta[3])
se[4] = se[4] * jnp.exp(beta[4])
#se[7] = se[7] * jnp.exp(beta[7])

result = np.array([beta, se])


np.savetxt('result.csv', result, delimiter=',')

'''
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

alpha = np.exp(np.dot(A_current, np.array([r[8], r[9], r[10], r[11]]))
                    + r[12]
                    )

beta = result[0]
se = result[1]

b1 = jnp.array([beta[0], beta[1], beta[2]])
b2 = jnp.array([beta[3],beta[4], beta[5], beta[6]])
c_o = beta[7]
b4 = jnp.array([beta[8], beta[9], beta[10], beta[11]])
c_alpha = beta[12]
rh=beta[13]
sigma_eta = beta[14]
sigma_nu = beta[15]
    #b8_1 = 2.21E-04
    #b8_2= -4.18E-04
    #b9_1 = 2.98E-03
b8 = jnp.array([2.21E-04, -4.18E-04])
b9 = jnp.array([2.98E-03])
c_i = 3.85E-01
#alpha = jnp.exp(jnp.dot(A_current, b4)
 #                   + c_alpha)

temp = demand_2018_using[(demand_2018_using['prem_id'] == 2.201020e+05) & (demand_2018_using['bill_ym'] == 201807.0)]

a_current_outdoor = [temp['heavy_water_spa_area'], temp['lawn_areaxNDVI'], temp['above_one_acre']]

a_current_indoor = [temp['bathroom'], temp['above_one_acre']]

a_current = [temp['heavy_water_spa'], temp['lawn_areaxNDVI'], temp['bathroom'], temp['above_one_acre']]

z_current_outdoor = [temp['mean_TMAX_1'], temp['lawn_areaxTmax'], temp['total_PRCP'], temp['lawn_areaxPRCP']]

z_current_indoor = [temp['mean_TMAX_1']]

i = [temp['income']]

small_p0 = [temp['previous_essential_usage_mp']]

p=3.1

fc_l = jnp.array([7.25+1.25, 7.25+3.55, 7.25+9.25, 7.25+29.75, 7.10+29.75])
p_l0 = jnp.array([2.89+0.2, 4.81+0.2, 8.34+0.2, 12.70+0.2, 14.21+0.2]) 
target_index = jnp.where(p_l0 - p <= 0)[0][-1]
p_l = p_l0.at[target_index].set(p)
q_kink_l = jnp.array([2, 6, 11, 20])
p_plus1_l = jnp.append(p_l[1:5],jnp.array([jnp.nan]) )
d_end = jnp.cumsum( (p_l - p_plus1_l)[:4] *q_kink_l)
d_end =  jnp.insert(d_end, 0, jnp.array([0.0]) )

p_k = p_l[0]

def calculate_dk (k):
    result = -fc_l[k] - d_end[k]
    return result

d_k = calculate_dk(0)

a_current = np.array(a_current).reshape((1, 4))

small_alpha = np.exp(np.dot(a_current, b4)+c_alpha )

rh = abs(rh)

a_current_outdoor = np.array(a_current_outdoor).reshape((1, 3))

a_current_indoor = np.array(a_current_indoor).reshape((1, 2))

z_current_outdoor = np.array(z_current_outdoor).reshape((1, 4))

z_current_indoor = np.array(z_current_indoor)

de = demand_2018['deflator'][2]
small_w_outdoor = np.exp(np.dot(a_current_outdoor, b1) + np.dot(z_current_outdoor, b2)
                #+ jnp.array(beta_3*G)
                - np.multiply(np.multiply(small_alpha,np.log(p_k)), de) + 
                np.multiply(rh, np.log(np.maximum(i+ np.multiply(d_k, de), 1e-16))) + c_o)
small_w_indoor = np.exp(np.dot(a_current_indoor, b8) 
                + z_current_indoor* b9
                + c_i
                )

'''

