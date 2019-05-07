#%% [markdown]
## Total passenger cleaning and imputation
#* Total passenger is one the more important columns as it is the basis for all of our targets 
#such as product take rate or data usage rate per flight 
#* There is roughly 3% missing data and about 0.1% of flights have more passengers than seats

#%%
import sys
import importlib
import re
import numpy as np
import pandas as pd

sys.path.append('python')


#%%
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 150) 


#%%
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


#%%
import wifipricing.data_reader
importlib.reload(wifipricing.data_reader)
import wifipricing.sugar 
importlib.reload(wifipricing.sugar)

from wifipricing.data_reader import data_reader
from wifipricing.sugar import *


#%%
# Seat count, total passenger issues
seatcol=[
    'aircraft_type',
    'airline',
    'seat_count',
    'total_passengers',
    'jack_seat_count',
]

# Sample random 1e6 rows
df = data_reader(
    "data/df_rfp_dataset_raw_20181218185047.csv",
    "data/data_reference.csv",
    skiprows=np.random.randint(1, 4000000, 3000000),  #random 1e6 samples
    usecols=seatcol
)


#%%
preview(df)
missing_data_report(df)


#%% [markdown]
# * On first glance, missing data isn't a huge issue with total passengers.
#  However, we get some strange flights that have more passengers than seats
#
# * It doesn't happen too often, but we should check seat count
# * Looks like we should ignore jack_seat_count

#%%
df['pass_seat_ratio'] = df['total_passengers'] / df['seat_count']
df['pass_jack_ratio'] = df['total_passengers'] / df['jack_seat_count']

f"More total_passengers than seat_count: {df[df['pass_seat_ratio'] > 1].shape[0] /  df.shape[0]:.2%}"
f"More total_passengers than jack_seat_count: {df[df['pass_jack_ratio'] > 1].shape[0] /  df.shape[0]:.2%}"

#%% [markdown]
# ## Compare flights that have strange passenger/seat ratio to the median of that flight
# * Looks like both total_passengers and seat_count can get strange values

#%%
problem_ac = distinct(df[df['pass_seat_ratio'] > 1], ['aircraft_type']) 

grp_means = pd.merge(problem_ac, df, how='inner').\
    loc[:, ['aircraft_type', 'total_passengers', 'seat_count']].dropna().\
    groupby('aircraft_type').agg(lambda x: x.dropna().median()).rename(columns=lambda x: x+'_median').reset_index()

prob = distinct(df[df['pass_seat_ratio'] > 1], ['aircraft_type', 'total_passengers', 'seat_count']) 
merged = pd.merge(prob, grp_means)
merged[merged.total_passengers_median > merged.seat_count_median]

#%% [markdown]
# #### Same as above, but comparison with airline/aircraft type median 
#%%
ac_al = distinct(df[df['pass_seat_ratio'] > 1], ['aircraft_type', 'airline']) 

grp_means = pd.merge(ac_al, df, how='inner').\
    loc[:, ['aircraft_type', 'airline', 'total_passengers', 'seat_count']].dropna().\
    groupby(['aircraft_type', 'airline']).median().rename(columns=lambda x: x+'_median').reset_index()

prob = distinct(df[df['pass_seat_ratio'] > 1], ['aircraft_type', 'airline', 'total_passengers', 'seat_count']) 
merged = pd.merge(prob, grp_means)
merged[merged.total_passengers_median > merged.seat_count_median]

#%% [markdown]
# ## We will change suspect total passenger/seats to median of aircraft type only 


#%% [markdown]
# ## Example
# ### Internal function to be used to data reader

#%% 
def _clean_passenger_seat(df):
    """
    NOTE: MODIFYING INPUT df  
    Imputing and fixing suspect total_passengers and seat_counts

    ----------------------------
    1. Missing values: both total_passengers and seat_count will get imputed by the
    median value for that airline + aircraft type 
    2. After imputation (no more missing data), replace entries where 
    total passengers > seat counts by the median of that aircraft type
    """
    req_cols = ['aircraft_type', 'airline', 'total_passengers', 'seat_count']
    assert set(req_cols).issubset(set(df.columns)), f"Input is missing one: {req_cols}"

    print('\nData cleaning on total_passenger and seat_count columns:')
    print('------------------------------------------------------')
    print('Input: missing data on total_passengers: {df.total_passengers.count()/df.shape[0]:.2%}')
    print(f"Flights with more passengers than seats: {sum(df['total_passengers']/df['seat_count'] > 1) / df.shape[0]:.2%}")

    print('\nimputing total passengers and seat counts by grouped median...')
    grp_acal = df.groupby(['aircraft_type', 'airline'])
    gm_acal = grp_acal.transform(lambda x: x.dropna().median())

    df.loc[:, 'total_passengers'] = df['total_passengers'].fillna(gm_acal['total_passengers'])
    df.loc[:, 'seat_count'] = df['seat_count'].fillna(gm_acal['seat_count'])

    print('Modifying suspect passengers and seat counts by grouped median...')
    grp_ac = df.groupby(['aircraft_type'])
    gm_ac = grp_ac.transform(lambda x: x.dropna().median())
    
    ind = (df['total_passengers'] > df['seat_count'])
    df.loc[ind, 'total_passengers'] = gm_ac.loc[ind, 'total_passengers']
    df.loc[ind, 'seat_count'] = gm_ac.loc[ind, 'seat_count']

    print('\nImputed: missing data on total_passengers: {df.total_passengers.count()/df.shape[0]:.2%}')
    print(f"Flights with more passengers than seats: {sum(df['total_passengers']/df['seat_count'] > 1) / df.shape[0]:.2%}")
    return df


#%%
clean_df = clean_passenger_seat(df[['aircraft_type', 'airline', 'total_passengers', 'seat_count']].copy(deep=True))

preview(clean_df)
