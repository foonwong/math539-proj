import pandas as pd
import re

def data_reader(data, data_dict, nrows=None, usecols=None):
    """reading in wifi data as pandas.DataFrame"""

    ref = pd.read_csv(data_dict)
    type_dict = {k:v for k,v in zip(ref['col_name'], ref['pandas_dtype'])}

    df = pd.read_csv(data, dtype=type_dict, nrows=nrows, usecols=usecols)

    timecols = [x for x in df.columns if 'Time' in x]
    df[timecols] = df[timecols].astype('datetime64')

    if 'ProductName' in df.columns: 
        prod_nm = df['ProductName'].unique()
        datacap_dict = {x:get_datacap(x) for x in prod_nm}
        timecap_dict = {x:get_timecap(x) for x in prod_nm}

        df['DataCap_MB'] = df['ProductName'].map(lambda x: datacap_dict[x])
        df['TimeCap_min'] = df['ProductName'].map(lambda x: timecap_dict[x])
    
    try:
        tr_df = get_takerate_overall(df)
        df = pd.merge(df, tr_df, on=['FlightID', 'TotalPassengers'], how='left')
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