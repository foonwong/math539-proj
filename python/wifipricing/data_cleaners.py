import pandas as pd
from itertools import compress
# from .sugar import *


def clean_wifi_data(df):
    print(f'\nData cleaning\n------------------------------------')
    try:
        _clean_passenger_seat(df)
    except:
        pass

    try:
        _clean_flight_type(df)
    except:
        pass

    try:
        _default_booleans(df)
    except:
        pass

    try:
        _clean_redeye(df)
    except:
        pass

    return None


def _clean_passenger_seat(df):
    """
    Imputing and fixing suspect total_passengers and seat_counts
    ----------------------------
    1. Missing values: both total_passengers and seat_count will get imputed by the
    median value for that airline + aircraft type 
    2. After imputation (no more missing data), replace entries where 
    total passengers > seat counts by the median of that aircraft type
    """
    req_cols = ['aircraft_type', 'airline', 'total_passengers', 'seat_count']
    assert set(req_cols).issubset(set(df.columns)), f"Input is missing one: {req_cols}"

    print(f"* total_passengers and seat_count:")
    print(f"\nInput:")
    _missing_message(df, 'total_passengers')
    print(f"Flights with more passengers than seats: {sum(df['total_passengers']/df['seat_count'] > 1) / df.shape[0]:.2%}")

    print('\nimputing total passengers and seat counts by aircraft_type+airline median...')
    grp_acal = df.groupby(['aircraft_type', 'airline'])
    gm_acal = grp_acal.transform(lambda x: x.dropna().median())

    df.loc[:, 'total_passengers'] = df['total_passengers'].fillna(gm_acal['total_passengers'])
    df.loc[:, 'seat_count'] = df['seat_count'].fillna(gm_acal['seat_count'])

    print('Modifying suspect passengers and seat counts by aircraft_type median...')
    grp_ac = df.groupby(['aircraft_type'])
    gm_ac = grp_ac.transform(lambda x: x.dropna().median())
    
    ind = (df['total_passengers'] > df['seat_count'])
    df.loc[ind, 'total_passengers'] = gm_ac.loc[ind, 'total_passengers']
    df.loc[ind, 'seat_count'] = gm_ac.loc[ind, 'seat_count']

    print(f"\nCleaned:")
    _missing_message(df, 'total_passengers')
    print(f"Flights with more passengers than seats: {sum(df['total_passengers']/df['seat_count'] > 1) / df.shape[0]:.2%}")
    return None


def _clean_flight_type(df):
    req_cols = ['orig_country', 'dest_country', 'flight_type']
    assert set(req_cols).issubset(set(df.columns)), f"Input is missing one of: {req_cols}"

    mis = df['flight_type'].isna()

    print(f"* total_passengers and seat_count:")
    print(f"\nInput:")
    _missing_message(df, 'flight_type')

    print(f"\n* Imputing flight_type base on orig/dest country:")

    is_dom = df.loc[mis, 'orig_country'].astype('object') == df.loc[mis, 'dest_country'].astype('object')
    df.loc[mis, 'flight_type'] = ['Domestic' if x else 'International' for x in list(is_dom)]

    print(f"\nCleaned:")
    _missing_message(df, 'flight_type')
    return None


def _clean_redeye(df):
    print("\n* Converting night_flight: {'RedEye':True, 'Normal':False}. Default to False") 

    _missing_message(df, 'night_flight')
    df['night_flight'] = [True if x else False for x in df['night_flight'].fillna('Normal') == 'RedEye']

    return None


def _default_booleans(df):
    bool_cols = ['ife', 'e_xtv', 'e_xphone', 'one_media', 'luxury']
    target_cols = [x for x in bool_cols if x in df.columns]

    print(f"\n* Imputing all missing data in the following columns to 0 and convert column to boolean")
    for col in target_cols:
        _missing_message(df, col)
        df[col] = df[col].fillna(0).astype('bool')

    return None


def _missing_message(df, col):
    "get proprtion of missing data for a particular column"
    mis = 1 - df[col].count() / df.shape[0]
    print(f"{col} missing: {mis:.4%}")
    return 