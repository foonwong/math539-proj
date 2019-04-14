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
        tr_df = get_takerate_overall(df)
        df = pd.merge(df, tr_df, on=['flight_id', 'total_passengers'], how='left')
    except:
        pass

    # try:
    #     tr_df = get_takerate_category(df)
    #     df = pd.merge(df, tr_df, on=['FlightID', 'TotalPassengers', 'Category'], how='left')
    # except:
    #     pass

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
        groupby(['FlightID', 'TotalPassengers']).size().reset_index().rename(columns={0:'TotalSessions'}).\
        assign(TakeRate_overall=lambda x: x['TotalSessions'] / x['TotalPassengers'])

    return tr_df


def get_takerate_category(df):
    """Returns overall wifi takerate per FlightID"""
    tr_df = df.\
        groupby(['FlightID', 'TotalPassengers', 'Category']).size().reset_index().rename(columns={0:'TotalSessionsCat'}).\
        assign(TakeRate_overall=lambda x: x['TotalSessionsCat'] / x['TotalPassengers'])

    return tr_df