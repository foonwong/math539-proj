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
from wifipricing.data_reader import get_flight_summary
from wifipricing.data_reader import get_product_summary
from wifipricing.sugar import *


#%%
# Seat count, total passenger issues
seatcol=[
    'aircraft_type',
    'seat_count',
    'total_passengers',
    'ac_frame',
    'jack_seat_count',
]

# Sample random 1e6 rows
df = data_reader(
    "data/df_rfp_dataset_raw_20181218185047.csv",
    "data/data_reference.csv",
    skiprows=np.random.randint(1, 40000000, 35000000),
    usecols=seatcol
)


#%%
df.head()
f'df dimensions: {df.shape}'
missing_data_report(df)


#%% [Markdown]
# * On first glance, missing data isn't a huge issue with total passengers 
# * However, we get some strange flights that have more passengers than seats
# It doesn't happen too often, but we should check seat count

#%%
df['pass_seat_ratio'] = df['total_passengers'] / df['seat_count']
df['pass_jack_ratio'] = df['total_passengers'] / df['jack_seat_count']


#%%
f"Too little seats: {df[df['pass_seat_ratio'] > 1].shape[0] /  df.shape[0]}"
f"Too little jack seats: {df[df['pass_jack_ratio'] > 1].shape[0] /  df.shape[0]}"

# Looks like we should ignore jack_seat_count

#%%
# Trying to see if we are getting strange passenger or seat count numbers

problem_ac = distinct(df[df['pass_seat_ratio'] > 1], ['aircraft_type']) 

mean_pass_seat = pd.merge(problem_ac, df, how='inner').\
    loc[:, ['aircraft_type', 'total_passengers', 'seat_count']].dropna().\
    groupby('aircraft_type').mean().rename(columns=lambda x: x+'_mean').reset_index()

mean_pass_seat

#%%
prob = distinct(df[df['pass_seat_ratio'] > 1], ['aircraft_type', 'total_passengers', 'seat_count']) 
pd.merge(prob, mean_pass_seat)


# Looks like both total_passengers and seat_count can get strange values


#%% [Markdown]

seatcol=[
    'aircraft_type',
    'seat_count',
    'total_passengers'
]

# Sample random 1e6 rows
df = data_reader(
    "data/df_rfp_dataset_raw_20181218185047.csv",
    "data/data_reference.csv",
    skiprows=np.random.randint(1, 40000000, 35000000),
    usecols=seatcol
)

#%%
def clean_passenger_seat(df):
    """
    Imputing and fixing suspect total_passengers and seat_counts

    ----------------------------
    1. Missing values: both total_passengers and seat_count will get imputed by the
    average value for that aircraft type 
    2. After imputation (no more missing data), replace entries where 
    total passengers > seat counts by the averages of that type
    """
    req_cols = ['aircraft_type', 'total_passengers', 'seat_count']
    assert set(req_cols).issubset(set(df.columns)), f"Input is missing one: {req_cols}"

    grouped = df.groupby('aircraft_type')
    gm_p = grouped['total_passengers'].transform('mean')
    gm_s = grouped['seat_count'].transform('mean')

    df['total_passengers'] = df['total_passengers'].fillna(gm_p)
    df['seat_count'] = df['seat_count'].fillna(gm_p)

    ind = (df['total_passengers'] / df['seat_count'] > 1)
    df['total_passengers'][ind] = gm_p[ind]
    df['seat_count'][ind] = gm_s[ind]

    return df



#%%
seatcol=[
    # 'product_name',
    # 'flight_id',
    # 'airline',
    'aircraft_type',
    # 'flight_duration_hrs',
    # 'total_usage_mb',
    'seat_count',
    'total_passengers',
    # 'night_flight',
    # 'flight_type',
    # 'category',
    'ac_frame',
    # 'routes',
    # 'price_usd',
    # 'ife',
    # 'e_xtv',
    # 'e_xphone',
    # 'one_media',
    'jack_seat_count',
    # 'economy_pct',
    # 'bus_pass_percent',
    # 'luxury',
    # 'origin_iata',
    # 'orig_country',
    # 'orig_region',
    # 'destination_iata',
    # 'dest_country',
    # 'dest_region'
    # 'airline_region'

    # 'antenna_type',
    # 'flight_duration_type'
    # 'session_start_time',
    # 'session_end_time',
    # 'session_duration_minutes',
    # 'session_volume_up_amount',
    # 'session_volume_down_amount',
    # 'departure_time_utc',
    # 'landing_time_utc',
    # 'departure_time_local',
    # 'landing_time_local'
]

#%%
