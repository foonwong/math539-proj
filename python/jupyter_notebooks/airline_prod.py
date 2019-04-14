#%%
import os
# sys.path.append('python')

os.listdir('python')
os.listdir('python/wifipricing')

#%%
import importlib
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
importlib.reload(wifipricing.data_reader)
import wifipricing.data_reader
from wifipricing.data_reader import data_reader
from wifipricing.data_reader import get_flight_summary
from wifipricing.data_reader import get_product_summary

sns.set(color_codes=True)

#%%
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 100) 

#%%
colnames_wifi = data_reader(
    "data/df_rfp_dataset_raw_20181218185047.csv",
    "data/data_reference.csv",
    nrows=5
).columns

colnames_wifi


#%%
df_price_cap = data_reader(
    "data/df_rfp_dataset_raw_20181218185047.csv",
    "data/data_reference.csv",
    usecols=['flight_id', 'routes', 'airline', 'price_usd', 'product_name', 'total_usage_mb']
)

df_price_cap = df_price_cap.\
    assign(price_per_mb = lambda x: x['datacap_mb']/ x['price_usd'])


#%%
df_price_cap

#%%
p_price_data = sns.jointplot(
    x='price_usd', y='total_usage_mb', 
    kind='kde',
    xlim=(0,40),
    ylim=(0,600),
    # color='DataCap_MB', kind='hex',
    data=df_price_cap.sample(10000)
 )

#%%
prop_1gb = round(sum(df_price_cap.total_usage_mb > 1000) / df_price_cap.shape[0] * 100, 
    ndigits=2)

print(f"Only {prop_1gb}% of customers use more than 1GB of data ")

#%%
price_levels = df_price_cap.groupby(['price_usd', 'product_name']).size().shape[0]
data_levels = df_price_cap.groupby(['price_usd', 'datacap_mb']).size().shape[0]
capped_perc = round(data_levels/price_levels * 100, 2)

not_nulls = df_price_cap.count()['datacap_mb']

capped_ses = round(not_nulls / df_price_cap.shape[0] *100, 2)

print(f'{data_levels} / {price_levels} ({capped_perc}%) of price/product combo have datacap')
print(f'{not_nulls} / {df_price_cap.shape[0]} ({capped_ses}%) of sessions have datacap')


print(f"Only {prop_1gb}% of customers use more than 1GB of data ")

#%% [markdown]
## Distributions

#%%
p = sns.distplot(df_price_cap['price_per_mb'].dropna(), )
p.axes.set_title('Distribution of price per MB')

#%%
df_price_cap.groupby('airline').size().sort_values().plot(kind='bar')

#%%
# number of products per airline-route
p = df_price_cap.groupby(['airline', 'routes', 'product_name']).size().\
    reset_index().groupby(['airline', 'routes']).size().plot(kind='hist')

p.axes.set_title('Number of products per airline-route (All)')

#%%
df_price_cap.query('datacap_mb > 0').\
    groupby(['airline', 'routes', 'product_name']).size().\
    reset_index().groupby(['airline', 'routes']).size().plot(kind='hist')

p.axes.set_title('Number of products per airline-route (Product with datacap)')

#%%
df_price_cap.query('timecap_min > 0').\
    groupby(['airline', 'routes', 'product_name']).size().\
    reset_index().groupby(['airline', 'routes']).size().plot(kind='hist')

p.axes.set_title('Number of products per airline-route (Product with timecap)')


#%%
df_price_cap.query('datacap_mb > 0').\
    groupby(['airline', 'routes', 'product_name']).size()


#%%
sns.jointplot(
    x='price_usd', y='datacap_mb', 
    kind='kde',
    data=df_price_cap.sample(10000)
 )


#%%
sns.jointplot(
    x='price_usd', y='datacap_mb', 
    kind='kde',
    xlim=(0,25),
    ylim=(0,150),
    # color='datacap_mb', kind='hex',
    data=df_price_cap.sample(10000)
 )

#%%
p = sns.jointplot(
    x='price_usd', y='price_per_mb', 
    kind='kde',
    data=df_price_cap.sample(10000)
 )

p.fig.suptitle('Distribution of product pricing vs price per MB')


#%%
p = sns.jointplot(
    x='price_per_mb', y='total_usage_mb', 
    kind='kde',
    xlim=(0,15),
    ylim=(0,150),
    data=df_price_cap.sample(10000)
 )

p.fig.suptitle('Distribution of data usage vs price per MB')

#%%
p = sns.jointplot(
    x='datacap_mb', y='total_usage_mb', 
    kind='kde',
    xlim=(0,150),
    ylim=(0,150),
    data=df_price_cap.sample(10000)
 )

p.fig.suptitle('Distribution of data usage vs price per MB')

#%%
p = sns.scatterplot(
    x=df_price_cap['datacap_mb'], y=df_price_cap['total_usage_mb'],
    # xlim=(0,150),
    # ylim=(0,150)
 )

p.set(xlim=(0,150), ylim=(0,150))
plt.show()


#%% [markdown]
## Distributions
### Considering where to limit training data

#%%
over_50 = df_price_cap.query('price_usd > 50').shape[0]
over_25 = df_price_cap.query('price_usd > 25').shape[0]
rows = df_price_cap.shape[0]

print(f'{over_50} / {rows} ({over_50/rows*100}%) of products are over $50 USD') 
print(f'{over_25} / {rows} ({over_25/rows*100}%) of products are over $25 USD') 

#%%
df_price_cap.columns


#%%
importlib.reload(wifipricing.data_reader)
from wifipricing.data_reader import data_reader
from wifipricing.data_reader import get_flight_summary
from wifipricing.data_reader import get_product_summary

df_test = data_reader(
    "data/df_rfp_dataset_raw_20181218185047.csv",
    "data/data_reference.csv",
    usecols=['flight_id', 'product_name', 'price_usd', 'total_usage_mb'],
    nrows=15
)

#%%
df_test


#%%
get_product_summary(df_test).shape


#%%


#%%
