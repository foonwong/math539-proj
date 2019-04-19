#%% [markdown]
## Making plots 

#%%
# Notebook options
import sys
import importlib
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('python')
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 15) 
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#%% [markdown]
### Pre-summarize data
#%%
# import wifipricing.modeling_prepartions
# importlib.reload(wifipricing.modeling_prepartions)
# from wifipricing.modeling_prepartions import get_splitted_wifi_data 
# df_data, df_time, df_none, df_all = get_splitted_wifi_data(
#     'data/df_rfp_dataset_raw_20181218185047.csv', 
#     'data/data_reference.csv'
# )

# df_all.columns
# df_all

#%%
df_data = pd.read_csv('data/summarized_data/summarized_datacap.csv')
df_time = pd.read_csv('data/summarized_data/summarized_timecap.csv')
df_all = pd.read_csv('data/summarized_data/summarized_all.csv')

f'Summarized (data-capped): {df_data.shape}'
f'Summarized (time-capped): {df_time.shape}'
f'Summarized (full dataset): {df_all.shape}'

#%% [markdown]
### Choosing columns to show on slides
#%%
df_all[df_all['datacap'] & df_all['timecap']]\
    .sample(n=30)\
    .filter(regex='product|usd|p.+count|total|_mb|_min|_per_')

df_all[df_all['datacap'] & df_all['timecap']]\
    .sample(n=30)\
    .filter(regex='product|cap')


#%%
smp_df_all = df_all.sample(n=5000)
sns.jointplot("revenue_per_psn", "data_per_psn", data=smp_df_all)


#%%
smp_df_all = df_all.groupby(['datacap', 'timecap'])\
    .apply(lambda x: x.sample(n=2000)).reset_index(drop=True)


g = sns.FacetGrid(df_all, row='datacap', col='timecap', 
    sharex=False, sharey=False,
    aspect=2, size=4)


g.map(sns.distplot, "profit_per_psn")


g = sns.FacetGrid(df_all.sample(10000), row='datacap', col='timecap', 
    sharex=False, sharey=False,
    aspect=2, height=4)

g.map(sns.kdeplot, "revenue_per_psn", "data_per_psn")


#%%
p = sns.distplot(smp_df_all.data_per_psn)
p.set_title('Distribution of data consumption per person, product, flight')
p.set_xlabel('data consumption (MB)')

#%%
p = sns.distplot(smp_df_all.revenue_per_psn)
p.set_title('Distribution of revenue per person, product, flight')
p.set_xlabel('revenue ($)')

#%%
p = sns.distplot(smp_df_all.price_usd)
p.set_title('Distribution of product pricing')
p.set_xlabel('Price ($)')

#%%
p = sns.scatterplot(smp_df_all.price_usd, smp_df_all.profit_per_psn, hue=smp_df_all.datacap_mb)
p.set_title('Profitability vs pricing')
p.set_ylabel('Profit per person ($)')
p.set_xlabel('Product pricing ($)')