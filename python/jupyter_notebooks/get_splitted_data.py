#%%
import sys
import os
import importlib
import re
import numpy as np
import pandas as pd

sys.path.append('python')
from wifipricing.data_reader import data_reader
from wifipricing.modeling_prepartions import get_splitted_wifi_data 

#%%
# returns a dict
dfs_summarized = get_splitted_wifi_data(
    'data/df_rfp_dataset_raw_20181218185047.csv', 
    'data/data_reference.csv'
)

#%%
for nm, df in dfs_summarized.items():
    df.to_pickle(f'data/summarized_data/df_summarized_{nm}_pickle.gzip')
