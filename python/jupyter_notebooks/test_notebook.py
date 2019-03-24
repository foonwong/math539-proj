#%%
sys.path.append('python')

#%%
import importlib
import re
import numpy as np
import pandas as pd
import wifipricing.data_reader
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 20) 

#%%
importlib.reload(wifipricing.data_reader)
from wifipricing.data_reader import data_reader

wifi_takerate_df = data_reader(
    "data/df_rfp_dataset_raw_20181218185047.csv",
    "data/data_reference.csv",
    usecols=['FlightID', 'TotalPassengers', 'Category'],
    nrows=5000
) 

df = wifi_takerate_df[['FlightID', 'TotalPassengers', 'Category']].head(100).\
    groupby(['FlightID', 'Category']).count().\
    unstack()

df.columns


#%%
df.head()

#%%
pd.merge(wifi_takerate_df, tr_df)


#%%

pd.merge(wifi_takerate_df, tr_df, on='FlightID').groupby(['FlightID', 'TakeRate_overall']).size()

#%%


#%%
