#%%
sys.path.append('python')

#%%
import importlib
import re
import numpy as np
import pandas as pd
import wifipricing.data_reader
importlib.reload(wifipricing.data_reader)
from wifipricing.data_reader import data_reader

# #%%
# wifi_products_df = data_reader(
#     "data/df_rfp_dataset_raw_20181218185047.csv",
#     "data/data_reference.csv",
#     usecols=['ProductName']
# )

# #%%
# wifi_products_df.astype(str). \
#     groupby(['ProductName', 'DataCap_MB', 'TimeCap_min']). \
#     size()


#%%
wifi_takerate_df = data_reader(
    "data/df_rfp_dataset_raw_20181218185047.csv",
    "data/data_reference.csv",
    usecols=['FlightID', 'TotalPassengers'],
    nrows=5000
) 

wifi_takerate_df.head()

#%%
tr_df.head()

#%%
pd.merge(wifi_takerate_df, tr_df, )


#%%

pd.merge(wifi_takerate_df, tr_df, on='FlightID').groupby(['FlightID', 'TakeRate_overall']).size()

#%%

#%%
