#%%
sys.path.append('python')

#%%
import importlib
import re
import numpy as np
import pandas as pd
import wifipricing.data_reader
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes=True)
e
#%%
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 10) 

#%%
colnames_wifi = data_reader(
    "data/df_rfp_dataset_raw_20181218185047.csv",
    "data/data_reference.csv",
    nrows=5
).columns

#%%
colnames_wifi

#%%
importlib.reload(wifipricing.data_reader)
from wifipricing.data_reader import data_reader

# df_airline_prod = data_reader(
#     "data/df_rfp_dataset_raw_20181218185047.csv",
#     "data/data_reference.csv",
#     usecols=['Price/USD', 'Airline', 'ProductName']
# ) 

# df_airline_prod.head(10)

# #%%
# df_airline_prod.groupby(['Airline', 'ProductName']).\
#     agg('count')

# #%%
# df_airline_prod.groupby(['ProductName']).\
#     agg('count').reset_index().\
#     query('ProductName')

#%%
df_price_cap = data_reader(
    "data/df_rfp_dataset_raw_20181218185047.csv",
    "data/data_reference.csv",
    usecols=['Price/USD', 'ProductName', 'TotalUsageMB']
) 


#%%
df_price_cap

#%%
p_price_data = sns.jointplot(
    x='Price/USD', y='TotalUsageMB', 
    kind='kde',
    xlim=(0,40),
    ylim=(0,600),
    # color='DataCap_MB', kind='hex',
    data=df_price_cap.sample(10000)
 )

#%%
# How many people are using crazy amount of data?
prop_1gb = round(sum(df_price_cap.TotalUsageMB > 1000) / df_price_cap.shape[0] * 100, 
    ndigits=2)

print(f"Only {prop_1gb}% of customers use more than 1GB of data ")

#%%
