import pandas as pd
import re
from itertools import compress
from sklearn.preprocessing import StandardScaler
from .data_reader import data_reader
from .data_reader import _flight_df_add_features
from .wifi_calculations import get_profit
from .sugar import missing_data_report

def get_lgb_df(summarized_df):

    return None



def get_splitted_wifi_data(wifi_path, ref_path, nrows=None):
    input_col=['product_name', 'flight_id', 'airline', 'aircraft_type',
        'flight_duration_hrs', 'total_usage_mb', 'seat_count', 'total_passengers',
        'night_flight', 'flight_type', 'price_usd', 'ife', 'e_xtv', 'e_xphone',
        'one_media', 'jack_seat_count', 'economy_pct', 'bus_pass_percent',
        'luxury', 'orig_country', 'dest_country']

    df = data_reader(wifi_path, ref_path, cleaning=True, usecols=input_col, nrows=nrows)

    # split into three subset depending on whether data has datacap
    _check_capped_sessions(df)
    drop_cols = ['aircraft_type']
    df = df.drop(columns=drop_cols)

    df_datacap = df[df['datacap']].pipe(get_product_summary)
    df_timecap = df[df['timecap']].pipe(get_product_summary)

    nocapcol = [x for x in df.columns if ('cap' not in x) and ('price_per' not in x)]
    df_nocap= df[~(df['datacap'] | df['timecap'])][nocapcol].\
        pipe(get_product_summary)

    df_all = get_product_summary(df)

    return [df_datacap, df_timecap, df_nocap, df_all] 


def get_product_summary(df_in, drop_cols=True):
    """Summarizing wifi dataframe for a product level analysis

    Will create new columns for product data/time cap, pricing and profit
    if possible.

    Note:
    Input dataframe needs to have at least:
    ['flight_id', 'product_name', 'total_passengers', 'price_usd', 'price_per_mb', 'price_per_min']

    In addition: input dataframe should be split according to data/timecap. 

    Columns that are more detailed than flight-product level will be discarded.

    By default, columns with missing data are also dropped after summarizing.
    """

    # Check minimum requirements to make product level summary
    req_cols = ['flight_id', 'product_name', 'total_passengers', 
        'price_usd']

    missed = [(x not in df_in.columns) for x in req_cols]

    if sum(missed): 
        raise Exception(f'Input df must have {list(compress(req_cols, missed))}')

    print(f'\nPrepare for modeling data at product level...')
    print(f'Input dataframe dimensions: {df_in.shape}')

    # include other columns that do not conflict with product level summary
    other_cols = ['airline', 'flight_duration_type', 'tail_number',
	    'aircraft_type', 'origin_iata', 'destination_iata',
	    'departure_time_utc', 'landing_time_utc', 'departure_time_local',
	    'landing_time_local', 'flight_duration_hrs', 'antenna_type',
	    'orig_country', 'orig_region', 'seat_count', 'night_flight',
	    'flight_type', 'category', 'airline_region', 'ac_frame', 'routes',
	    'ife', 'e_xtv', 'e_xphone', 'one_media', 'dest_country', 'dest_region',
	    'jack_seat_count', 'economy_pct', 'bus_pass_percent', 'luxury']

    grp_cols = [x for x in df_in.columns if x in (req_cols + other_cols)] 

    df_out = df_in.groupby(grp_cols).size().reset_index().drop(columns=0)

    try:
        df_du = _get_data_per_psn(df_in)
        df_out = pd.merge(df_out, df_du, on=['flight_id', 'product_name', 'total_passengers'], how='left')
    except:
        pass

    try:
        df_rev = _get_rev_per_psn(df_in)
        df_out = pd.merge(df_out, df_rev, on=['flight_id', 'product_name'], how='left')
    except:
        pass

    try:
        df_out['profit_per_psn'] = get_profit(df_out['revenue_per_psn'], df_out['data_per_psn'])
    except:
        pass

    _flight_df_add_features(df_out)

    print('\n------------------\n Grouped data:')
    missing_data_report(df_out)

    if drop_cols:
        print('Columns with missing values will be dropped')
        df_out = df_out.dropna(axis='columns')
        print(f'Final dataframe dimensions: {df_out.shape}')

    return df_out


def _get_rev_per_psn(df):
    "Returns revenue per product per head"
    df_rev = df.groupby(['flight_id', 'product_name', 'total_passengers'])['price_usd'].sum().\
        reset_index().\
        assign(revenue_per_psn = lambda x: x['price_usd'] / x['total_passengers']).\
        drop(columns=['price_usd', 'total_passengers'])

    return df_rev


def _get_data_per_psn(df):
    """Returns overall datausage per product per head on a flight"""
    df_du = df.groupby(['flight_id', 'product_name', 'total_passengers'])['total_usage_mb'].sum().\
        reset_index().rename(columns={'total_usage_mb':'data_per_psn'}).\
        assign(data_per_psn = lambda x: x['data_per_psn'] / x['total_passengers'])

    return df_du


def _check_capped_sessions(df):
    print ('Sessions with caps:\n-------------------------------------')
    nrow = df.shape[0]
    datacapped = df[df['datacap']].shape[0]
    timecapped = df[df['timecap']].shape[0]
    bothcapped = df[df['datacap'] & df['timecap']].shape[0]
    nocapped = df[~(df['datacap'] | df['timecap'])].shape[0]

    print(f'Data: {datacapped} rows, {datacapped/nrow:.2%}')
    print(f'Time: {timecapped} rows, {timecapped/nrow:.2%}')
    print(f'Both: {bothcapped} rows, {bothcapped/nrow:.2%}')
    print(f'None: {nocapped} rows, {nocapped/nrow:.2%}')

    return None