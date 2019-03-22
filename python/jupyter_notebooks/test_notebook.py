#%%
sys.path.append('python/wifipricing')
for p in sys.path:
    print(p)


#%%
import re
import numpy as np
import data_reader
import importlib
importlib.reload(data_reader)

#%%
wifi_df = data_reader.data_reader(
    "data/df_rfp_dataset_raw_20181218185047.csv",
    "data/data_reference.csv",
    usecols=['ProductName'],
    nrows=100000
)

#%%
wifi_df

#%%
wifi_df.dtypes

#%%
wifi_df['ProductName'].map(lambda x: datacap[x]).unique()