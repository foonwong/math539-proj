import pandas as pd
import re
from itertools import compress

def data_reader(data, data_dict, nrows=None, skiprows=None, usecols=None):
    """reading in wifi data as pandas.DataFrame.

    Will create new columns for product data/time cap, pricing and profit
    if possible.

    Keyword arguments:
    data -- path to wifi csv
    data_dict -- path to reference csv with pandas colname and dtype info
    nrows -- pd.read_csv nrows. Use None for all rows. (default None)
    skiprows -- pd.read_csv skiprows. 
    usecols -- pd.read_csv usecols. Use None for all cols. (default None)
    """

    ref = pd.read_csv(data_dict)

    pd_nm = ref['pandas_name']
    type_dict = {k:v for k,v in zip(pd_nm, ref['pandas_dtype'])}

    df = pd.read_csv(
        data,
        names = pd_nm, header=1,
        nrows=nrows, skiprows = skiprows, usecols=usecols,
        dtype = type_dict
    )

    timecols = [x for x in df.columns if 'time' in x]
    df[timecols] = df[timecols].astype('datetime64')

    df = flight_df_add_features(df)

    return df


def flight_df_add_features(df):
    """Add extra features to wifi dataframe if possible"""
    try:
        prod_nm = df['product_name'].unique()
        datacap_dict = {x:get_datacap(x) for x in prod_nm}
        timecap_dict = {x:get_timecap(x) for x in prod_nm}

        df['datacap_mb'] = df['product_name'].map(lambda x: datacap_dict[x])
        df['timecap_min'] = df['product_name'].map(lambda x: timecap_dict[x])
    except:
        pass

    try:
        df['price_per_mb'] = df['price_usd'] / df['datacap_mb']
        df['price_per_min'] =  df['price_usd'] / df['timecap_min']
    except:
        pass

    try:
        df['profit'] = get_profit(df['price_usd'], df['total_usage_mb'])
    except:
        pass

    return df


def get_flight_summary(df_detailed):
    """Summarizing wifi dataframe for a flight level analysis"""
    df = df_detailed.groupby('flight_id').size().reset_index().drop(columns=0)

    try:
        df_tr = get_takerate_overall(df_detailed)
        df = pd.merge(df, df_tr, on=['flight_id', 'total_passengers'], how='left')
    except:
        pass

    try:
        df_du = get_flight_data_mb(df_detailed)
        df = pd.merge(df, df_du, on=['flight_id'], how='left')
    except:
        pass

    try:
        df_rev = get_flight_revenue(df_detailed)
        df = pd.merge(df, df_rev, on=['flight_id'], how='left')
    except:
        pass

    return df


def get_product_summary(df_in):
    """Summarizing wifi dataframe for a product level analysis

    Will create new columns for product data/time cap, pricing and profit
    if possible.

    Note:
    df_in needs to have at least:
    ['flight_id', 'product_name', 'total_passengers', 'price_usd', 'price_per_mb', 'price_per_min']

    Columns that are more detailed than flight-product level will be discarded
    """

    # Check minimum requirements to make product level summary
    # req_cols = ['flight_id', 'product_name', 'total_passengers', 
    #     'price_usd', 'price_per_mb', 'price_per_min']

    req_cols = ['flight_id', 'product_name', 'total_passengers', 
        'price_usd']

    missed = [(x not in df_in.columns) for x in req_cols]

    if sum(missed): 
        raise Exception(f'Input df must have {list(compress(req_cols, missed))}')


    # include other columns that do not conflict with product level summary
    other_cols = ['airline', 'flight_duration_type', 'tail_number',
	    'aircraft_type', 'origin_iata', 'destination_iata',
	    'departure_time_utc', 'landing_time_utc', 'departure_time_local',
	    'landing_time_local', 'flight_duration_hrs', 'antenna_type',
	    'orig_country', 'orig_region', 'seat_count', 'night_flight',
	    'flight_type', 'category', 'airline_region', 'ac_frame', 'routes',
	    'ife', 'e_xtv', 'e_xphone', 'one_media', 'dest_country', 'dest_region',
	    'jack_seat_count', 'economy_pct', 'bus_pass_percent', 'luxury']

    df = distinct(df_in, [x for x in df_in.columns if x in (req_cols + other_cols)])

    try:
        df_du = get_data_per_psn(df_in)
        df = pd.merge(df, df_du, on=['flight_id', 'product_name', 'total_passengers'], how='left')
    except:
        pass

    try:
        df_rev = get_rev_per_psn(df_in)
        df = pd.merge(df, df_rev, on=['flight_id', 'product_name'], how='left')
    except:
        pass

    try:
        df['profit_per_psn'] = get_profit(df['revenue_per_psn'], df['data_per_psn'])
    except:
        pass

    df = flight_df_add_features(df)

    return df


def get_datacap(x):
    """Extract data cap in MB from productname"""
    data = re.search('(?i)[\d]+(?= ?MB)', x)

    try:
        data =  float(data.group())
    except:
        data = None

    return data


def get_profit(revenue, data_mb, cost_per_mb = 0.05):
    """Calculate profit from revenue and data usage"""
    try:
        prof = revenue - data_mb * cost_per_mb
    except:
        prof = None

    return prof



def get_timecap(x):
    """Extract time cap in min from productname"""
    data = re.search('(?i)([\d]+) ?(min|h)', x)

    try:
        if data.group(2).upper() == 'H':
            data =  60 * float(data.group(1))
        else:
            data = float(data.group(1))
    except:
        data = None

    return data


def get_takerate_overall(df):
    """Returns overall wifi takerate per FlightID"""
    tr_df = df.\
        groupby(['flight_id', 'total_passengers']).size().reset_index().rename(columns={0:'totalsessions'}).\
        assign(flight_takerate = lambda x: x['total_sessions'] / x['total_passengers'])

    return tr_df


def get_flight_revenue(df):
    "Returns overall revenue per FlightID"
    df_rev = df.groupby('flight_id')['price_usd'].sum().\
        reset_index().rename(columns={'price_usd':'flight_revenue_usd'})

    return df_rev


def get_flight_data_mb(df):
    """Returns overall datausage per FlightID"""
    df_du = df.groupby('flight_id')['total_usage_mb'].sum().\
        reset_index().rename(columns={'total_usage_mb':'flight_data_mb'})

    return df_du


def get_rev_per_psn(df):
    "Returns revenue per product per head"
    df_rev = df.groupby(['flight_id', 'product_name', 'total_passengers'])['price_usd'].sum().\
        reset_index().\
        assign(revenue_per_psn = lambda x: x['price_usd'] / x['total_passengers']).\
        drop(columns=['price_usd', 'total_passengers'])

    return df_rev


def get_data_per_psn(df):
    """Returns overall datausage per product per head on a flight"""
    df_du = df.groupby(['flight_id', 'product_name', 'total_passengers'])['total_usage_mb'].sum().\
        reset_index().rename(columns={'total_usage_mb':'data_per_psn'}).\
        assign(data_per_psn = lambda x: x['data_per_psn'] / x['total_passengers'])

    return df_du


def distinct(df, cols):
    "similar to dplyr's distinct"

    df = df.groupby(cols).size().reset_index().\
        drop(columns=0)

    return df


def move_target_to_end(df, target):
    "Move target column to the last column"
    colorder = [col for col in df.columns if col != target]
    colorder.append(target)

    return df[colorder]



