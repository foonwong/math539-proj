#%% [markdown]
## Making plots 

#%%
# Notebook options
import os
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
paths = [p for p in os.listdir('data/summarized_data/') if 'pickle' in p]
df_all, df_data, df_time, _ = [pd.read_pickle(os.path.join('data/summarized_data', p)) for p in paths]

#%%
df_data
df_data.filter(regex='per_')


#%%
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
    aspect=2, size=8)


g.map(sns.distplot, "profit_per_psn")


# g = sns.FacetGrid(df_all.sample(10000), row='datacap', col='timecap', 
    # sharex=False, sharey=False,
    # aspect=2, height=4)
# 
# g.map(sns.kdeplot, "revenue_per_psn", "data_per_psn")

#%%
smp_df_data = df_data.sample(n=5000)

plt.figure(figsize=(6, 8))
fig = sns.distplot(smp_df_data['profit_per_psn'])
fig.set_title('Profit per person (datacap subset)')

plt.figure(figsize=(6, 8))
fig = sns.distplot(smp_df_all['profit_per_psn'])
fig.set_title('Profit per person (Full dataset)')


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