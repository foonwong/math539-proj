import pandas as pd
import re

def data_reader(data, data_dict, nrows=None, usecols=None):
    """reading in wifi data as pandas.DataFrame"""

    ref = pd.read_csv(data_dict)

    pd_nm = ref['pandas_name']
    type_dict = {k:v for k,v in zip(pd_nm, ref['pandas_dtype'])}

    # df = pd.read_csv(data, names = pd_nm, dtype=type_dict, nrows=nrows, usecols=usecols)
    df = pd.read_csv(
        data, 
        names = pd_nm, header=1, 
        nrows=nrows, usecols=usecols,
        dtype = type_dict
    )

    timecols = [x for x in df.columns if 'time' in x]
    df[timecols] = df[timecols].astype('datetime64')

    if 'product_name' in df.columns: 
        prod_nm = df['product_name'].unique()
        datacap_dict = {x:get_datacap(x) for x in prod_nm}
        timecap_dict = {x:get_timecap(x) for x in prod_nm}

        df['datacap_mb'] = df['product_name'].map(lambda x: datacap_dict[x])
        df['timecap_min'] = df['product_name'].map(lambda x: timecap_dict[x])
    
    try:
        df_tr = get_takerate_overall(df)
        df = pd.merge(df, df_tr, on=['flight_id', 'total_passengers'], how='left')
    except:
        pass

    try:
        df_du = get_flight_data_mb(df)
        df = pd.merge(df, df_du, on=['flight_id'], how='left')
    except:
        pass

    try:
        df_rev = get_revenue(df)
        df = pd.merge(df, df_rev, on=['flight_id'], how='left')
    except:
        pass

    return df 


def get_datacap(x):
    """Extract data cap in MB from productname"""
    data = re.search('(?i)[\d]+(?= ?MB)', x)
    
    try:
        data =  float(data.group())
    except:
        data = None

    return data


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


def get_revenue(df):
    "Returns overall revenue per FlightID"
    df_rev = df.groupby('flight_id')['price_usd'].sum().\
        reset_index().rename(columns={'price_usd':'flight_revenue_usd'})

    return df_rev

#%%
def get_flight_data_mb(df):
    """Returns overall datausage per FlightID"""
    df_du = df.groupby('flight_id')['total_usage_mb'].sum().\
        reset_index().rename(columns={'total_usage_mb':'flight_data_mb'})

    return df_du

# def get_takerate_category(df):
#     """Returns overall wifi takerate per FlightID"""
#     tr_df = df.\
#         groupby(['FlightID', 'TotalPassengers', 'Category']).size().reset_index().rename(columns={0:'TotalSessionsCat'}).\
#         assign(TakeRate_overall=lambda x: x['TotalSessionsCat'] / x['TotalPassengers'])

#     return tr_df