#%%
sys.path.append('python')


#%%
import importlib
import re
import numpy as np
import pandas as pd
import wifipricing.data_reader
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 100) 

#%%
importlib.reload(wifipricing.data_reader)
from wifipricing.data_reader import data_reader

df_airline_product = data_reader(
    "data/df_rfp_dataset_raw_20181218185047.csv",
    "data/data_reference.csv",
    usecols=['Airline', 'ProductName']
) 


#%%
# wifi_takerate_df[['FlightID', 'TotalPassengers', 'Category']].head(100).\
#     groupby(['FlightID', 'Category']).count().\
#     unstack()

wifi_takerate_df[['FlightID', 'TotalPassengers', 'Category']]\
    .groupby(['FlightID', 'Category']).size()\
    .unstack().fillna(0)
    # assign(total = lambda x: x['Browsing'] + x['Streaming'] + x['Text'])



#%%
df.head()

#%%
pd.merge(wifi_takerate_df, tr_df)


#%%

pd.merge(wifi_takerate_df, tr_df, on='FlightID').groupby(['FlightID', 'TakeRate_overall']).size()

#%%
df_airline_product.groupby(['Airline', 'ProductName']).\
    size().reset_index()

#%%
df_airline_product.groupby(['Airline', 'ProductName', 'DataCap_MB']). \
    agg('count')
    



#%%
