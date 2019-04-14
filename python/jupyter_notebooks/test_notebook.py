#%%
sys.path.append('python')

#%%
import importlib
import re
import numpy as np
import pandas as pd
import wifipricing.data_reader

#%%
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 10) 

#%%
importlib.reload(wifipricing.data_reader)
from wifipricing.data_reader import data_reader

df_wifi_takerate = data_reader(
    "data/df_rfp_dataset_raw_20181218185047.csv",
    "data/data_reference.csv",
    # usecols=['FlightID', 'TotalPassengers', 'Category'],
    usecols=['ProductName'],
    nrows=5000
) 

colnames_wifi = pd.DataFrame({'columns': data_reader(
    "data/df_rfp_dataset_raw_20181218185047.csv",
    "data/data_reference.csv",
    nrows=5
).columns})

#%%
colnames_wifi