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
from matplotlib.patches import Patch


price_elasticity_df = pd.read_csv('../price elasticity/price_elasticity_df.csv')

NDVI_cluster = pd.read_csv('../premise_segments_roster.csv')

demand_2018_using_new_small = pd.read_csv('../demand_2018_using_new_small.csv')


# 2. Prepare the Keys
# Ensure 'prem_id' is the same type in both (string is safest to avoid float/int mismatches)
price_elasticity_df['prem_id'] = price_elasticity_df['prem_id'].astype(str)
NDVI_cluster['prem_id'] = NDVI_cluster['prem_id'].astype(str)
demand_2018_using_new_small['prem_id'] = demand_2018_using_new_small['prem_id'].astype(str)

# 3. Perform the Merge
# We use a Left Join to keep every billing record from your panel data
# and attach the single cluster label to all of that household's bills.
merged_df = price_elasticity_df.merge(
    NDVI_cluster[['prem_id', 'label', 'cluster', 'mean_ndvi', 'corr_level']], 
    on='prem_id', 
    how='left'
)

merged_df = merged_df.merge(
    demand_2018_using_new_small[['prem_id', 'charge', 'lawn_area', 'mean_e_diff']], 
    on='prem_id', 
    how='left'
)

# 4. Validation & Quick Look
print(f"Original Panel Rows: {len(price_elasticity_df)}")
print(f"Merged Panel Rows:   {len(merged_df)}")
print("-" * 30)

# Check for unmatchable premises (NaN labels)
missing_count = merged_df['label'].isna().sum()
if missing_count > 0:
    print(f"WARNING: {missing_count} rows ({missing_count/len(merged_df):.1%}) did not match a cluster.")
else:
    print("SUCCESS: 100% of rows matched a cluster.")
# 5. The "Golden Insight" Check
# Group by your new clusters and see if they behave differently on price
print("\nAverage Price Elasticity by Cluster:")
print(merged_df.groupby('label')['price_elasticity'].mean().sort_values())


# 1. Prepare the Data (Aggregate to Household Level first)
# This removes monthly noise and gives equal weight to every house
hh_level_df = merged_df.groupby(['prem_id', 'label']).agg({
    'price_elasticity': 'mean',  # Average elasticity per HH
    'charge': 'mean',  # Average Payment
    'lawn_area': 'mean',  # Lawn Area is static, mean just grabs the value
    'mean_e_diff': 'mean',  # Mean eta is static, mean just grabs the value
    'income': 'mean',            # Income is static, mean just grabs the value
    'bedroom': 'max',            # Static feature
    'bathroom': 'max'            # Static feature
}).reset_index()

# 2. Fix Units (Scale Income to $1000s)
hh_level_df['income_k'] = hh_level_df['income'] / 1000

# 3. Define Metrics (Using the new scaled income)
# Note: We use 'income_k' now instead of 'income'
metrics = ['price_elasticity', 'income_k', 'charge', 'mean_e_diff','lawn_area','bathroom']

# --- PART 1: STATISTICAL SUMMARY ---
summary_table = hh_level_df.groupby('label')[metrics].agg(['mean', 'median', 'std'])
print("\n--- Economic & Property Profile (Per Household) ---")
print(summary_table.round(2))

# --- PART 2: VISUALIZATION DASHBOARD ---
fig, axes = plt.subplots(3, 2, figsize=(24, 16))
axes = axes.flatten()
plt.subplots_adjust(hspace=0.35, wspace=0.2)

palette_order = sorted(hh_level_df['label'].unique())
palette = dict(zip(
    palette_order,
    sns.color_palette("deep", len(palette_order))
))

# Plot A
sns.boxplot(x='label', y='price_elasticity', data=hh_level_df,
            ax=axes[0], palette=palette, order=palette_order, showfliers=False)
axes[0].set_title('Avg Price Elasticity (Household Level)')
axes[0].set_ylabel('Elasticity')
axes[0].axhline(0, color='black', linestyle='--')

# Plot B
sns.boxplot(x='label', y='income_k', data=hh_level_df,
            ax=axes[1], palette=palette, order=palette_order, showfliers=False)
axes[1].set_title('Household Income')
axes[1].set_ylabel('Income ($1000s)')

# Plot C
sns.boxplot(x='label', y='charge', data=hh_level_df,
            ax=axes[2], palette=palette, order=palette_order, showfliers=False)
axes[2].set_title('Payments')
axes[2].set_ylabel('$')

# Plot D
sns.boxplot(x='label', y='mean_e_diff', data=hh_level_df,
            ax=axes[3], palette=palette, order=palette_order, showfliers=False)
axes[3].set_title('Mean Eta')
axes[3].set_ylabel('η')

# Plot E
sns.boxplot(x='label', y='lawn_area', data=hh_level_df,
            ax=axes[4], palette=palette, order=palette_order, showfliers=False)
axes[4].set_title('Lawn Area')
axes[4].set_ylabel('sqft')

# Plot F
sns.boxplot(x='label', y='bathroom', data=hh_level_df,
            ax=axes[5], palette=palette, order=palette_order, showfliers=False)
axes[5].set_title('Bathroom')
axes[5].set_ylabel('#')

# Shared legend
legend_elements = [
    Patch(facecolor=palette[label], label=label)
    for label in palette_order
]

fig.legend(handles=legend_elements,
           loc='lower center',
           ncol=len(palette_order),
           fontsize=12,
           frameon=False)

short_labels = ['A', 'B', 'C', 'D']

# Tick cleanup
for ax in axes:
    ax.set_xticklabels(short_labels)
    ax.set_xlabel("")
    ax.tick_params(axis='x', labelsize=12)
    ax.title.set_weight('bold')

# Save
os.makedirs("plot", exist_ok=True)
plt.savefig("plot/cluster_dashboard.png",
            dpi=100,
            bbox_inches='tight')

plt.show()
