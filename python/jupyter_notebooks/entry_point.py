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
df_price_cap = data_reader(
    "data/df_rfp_dataset_raw_20181218185047.csv",
    "data/data_reference.csv",
    skiprows=lambda i: i % 1000 != 0, # only get 1/1000 rows
    usecols=['flight_id', 'product_name', 'total_passengers', 'price_usd',
        'night_flight']
)

#%%
get_product_summary(df_price_cap)