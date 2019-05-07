#%%
import sys
import os
import importlib
import re
import numpy as np
import pandas as pd

sys.path.append('python')


#%%
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 10) 


#%%
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


#%%
import wifipricing.data_reader
importlib.reload(wifipricing.data_reader)
import wifipricing.modeling_prepartions
importlib.reload(wifipricing.modeling_prepartions)
import wifipricing.sugar 
importlib.reload(wifipricing.sugar)

from wifipricing.data_reader import data_reader
from wifipricing.modeling_prepartions import get_splitted_wifi_data 
from wifipricing.sugar import *

#%%
# a dict
dfs_summarized = get_splitted_wifi_data(
    'data/df_rfp_dataset_raw_20181218185047.csv', 
    'data/data_reference.csv'
)

#%%

for nm, df in dfs_summarized.items():
    df.to_pickle(f'data/summarized_data/df_summarized_{nm}_pickle.gzip')