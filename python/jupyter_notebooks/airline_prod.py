#%%
sys.path.append('python')

#%%
import importlib
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes=True)

#%%
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 10) 


#%%
import wifipricing.data_reader
importlib.reload(wifipricing.data_reader)
from wifipricing.data_reader import data_reader
from wifipricing.data_reader import get_flight_summary
from wifipricing.data_reader import get_product_summary
from wifipricing.data_reader import distinct


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
    usecols=['flight_id', 'routes', 'airline', 'price_usd', 'total_passengers',
             'product_name', 'total_usage_mb'],
)

#%%
df_price_cap

#%%
df_price_cap.quantile(0.98)

#%%
prop_1gb = round(sum(df_price_cap.total_usage_mb > 1000) / df_price_cap.shape[0] * 100, 
    ndigits=2)

print(f"Only {prop_1gb}% of customers use more than 1GB of data ")


#%% [markdown]
# ## First get a sense of the distribution of product pricing/data usage
# 2. apply filter 
# 3. Make product summary

#%% [markdown]
# ## First get a sense of the distribution of product pricing/data usage

#%%
sns.pairplot(df_price_cap[['total_usage_mb', 'price_usd', 'datacap_mb', 
                           'timecap_min', 'price_per_mb', 'profit']].sample(1000))

#%%
sns.jointplot(
    x='price_usd', y='total_usage_mb', 
    kind='kde',
    xlim=(0,40),
    ylim=(0,900),
    data=df_price_cap.sample(10000)
 )


#%%
price_levels = df_price_cap.groupby(['price_usd', 'product_name']).size().shape[0]
data_levels = df_price_cap.groupby(['price_usd', 'datacap_mb']).size().shape[0]
capped_perc = round(data_levels/price_levels * 100, 2)

not_nulls = df_price_cap.count()['datacap_mb']

capped_ses = round(not_nulls / df_price_cap.shape[0] *100, 2)

print(f'{data_levels} / {price_levels} ({capped_perc}%) of price/product combo have datacap')
print(f'{not_nulls} / {df_price_cap.shape[0]} ({capped_ses}%) of sessions have datacap')


#%%
##Distributions
p = sns.distplot(df_price_cap['price_per_mb'].dropna(), )
p.axes.set_title('Distribution of price per MB')

#%%
##profit per session
sns.distplot(df_price_cap.query('total_usage_mb < 1200')['profit'])


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
# number of products per flight 
p = df_price_cap.groupby(['flight_id', 'product_name']).size().\
    reset_index().groupby(['flight_id']).size().plot(kind='hist', bins=6)

p.axes.set_title('Number of products per flight (All)')


#%%
# number of products per flight (with cap)
p = df_price_cap.query('datacap_mb > 0').\
    groupby(['flight_id', 'product_name']).size().\
    reset_index().groupby(['flight_id']).size().plot(kind='hist', bins=6)

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
    x='price_per_mb', y='profit', 
    kind='kde',
    xlim=(0, 30),
    ylim=(-10,30),
    data=df_price_cap.sample(10000)
 )

p.fig.suptitle('Distribution of price per MB vs profit per session')


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


#%%


#%%
def distinct(df, cols):
    "similar to dplyr's distinct"

    df = df.groupby(cols).size().reset_index().\
        drop(columns=0)

    return df



#%%
df_prod_summary = df_price_cap.\
    query('total_usage_mb < 1200').\
    query('price_per_mb < 25').\
    query('price_usd < 65').\
    pipe(get_product_summary)

#%%
df_prod_summary

#%%
prod_quantile = df_prod_summary.quantile(.99)
prod_quantile


#%%
sns.pairplot(df_prod_summary[prod_quantile.index].sample(5000))

#%%
sns.jointplot(
    x='datacap_mb', y='profit_per_psn',
    kind='kde',
    data=df_prod_summary.sample(50000)
)

#%%
# It doesn't happen often, but sometimes there are multiple prices for the
# same product per flight

hmm = distinct(df_prod_summary, ['flight_id', 'product_name', 'price_per_mb']).\
    groupby(['flight_id', 'product_name']).size().reset_index().\
    rename(columns={0:"counts"}).\
    sort_values(by=['counts'], ascending=False).\
    query('counts > 1')


#%%
pd.merge(hmm, df_price_cap, how='left')

#%%
